from pprint import pprint

from transformers import pipeline

from utils.hook import Hook


def test_hook():
    pipe = pipeline("text-generation", "openai-community/gpt2")

    c_attn_hooks: list[Hook] = [
        Hook(
            pipe.model.transformer.h[i].attn.c_attn,
        )
        for i in range(12)
    ]

    with Hook.context(c_attn_hooks):
        pipe("Hello, how are you?")
    print(c_attn_hooks[0].inputs)
    print(c_attn_hooks[0].outputs)

    # query_states = torch.stack([hook.result.query_states for hook in hooks])
    # key_states = torch.stack([hook.result.key_states for hook in hooks])
