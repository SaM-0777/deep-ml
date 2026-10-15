def kv_cache_memory(batch_size: int, seq_len: int, embed_dim: int, n_heads: int, n_layers: int, bytes_per_elem: int) -> list:
    """
    Return [mha_bytes, linear_bytes, mixed_bytes] for the three architectures.
    Mixed uses a 3:1 ratio of linear-attention layers to MHA layers.
    """
    d_heads = embed_dim // n_heads
    
    mha_bytes = 2 * batch_size * seq_len * n_heads * d_heads * bytes_per_elem

    gdn_bytes = batch_size * n_heads * d_heads * d_heads * bytes_per_elem

    total_mha = n_layers * mha_bytes
    total_gdn = n_layers * gdn_bytes

    mixed_bytes = 0.75 * n_layers * gdn_bytes + 0.25 * n_layers * mha_bytes

    return [total_mha, total_gdn, mixed_bytes]

