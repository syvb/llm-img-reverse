import torch
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionModel

from llm_img_reverse.vision import MODEL_ID, encode, to_pixel_values


def test_encode_matches_hf_forward():
    # A small randomly initialized tower with the real layout; encode reimplements its forward.
    config = Qwen3VLVisionConfig(
        depth=5, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=96,
        num_position_embeddings=64, deepstack_visual_indexes=[1, 3],
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3VLVisionModel(config).eval()
    image = torch.rand(3, 64, 96)

    main, deepstack = model(*to_pixel_values(image, config))
    tokens = encode(model, image)
    assert list(tokens) == ["main", "ds1", "ds3"]
    for actual, expected in zip(tokens.values(), [main, *deepstack]):
        torch.testing.assert_close(actual, expected)
    # Stopping early must not change the shallower streams.
    torch.testing.assert_close(encode(model, image, streams=["ds1"])["ds1"], deepstack[0])


def test_pixel_values_match_processor():
    config = AutoConfig.from_pretrained(MODEL_ID).vision_config
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # Non-square, with distinct rows and columns, so a transposed or misordered layout fails.
    image = torch.randint(0, 256, (3, 64, 96), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))

    expected = processor(images=[image], do_resize=False, return_tensors="pt")
    pixel_values, grid_thw = to_pixel_values(image.float() / 255, config)

    assert grid_thw.tolist() == expected["image_grid_thw"].tolist() == [[1, 4, 6]]
    torch.testing.assert_close(pixel_values, expected["pixel_values"], atol=1e-5, rtol=0)
