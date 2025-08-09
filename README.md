# f5_tts_infer_min
Minimal example of f5_tts inference with custom model

## Pain points
- the reference audio has to match the reference text exactly (what is said, the same has to be written)
- when you pass reference audio longer than 12s it will be trimmed to 12s - so reference text could not match
- when you pass reference text as empty string the reference audio will be transcribed automatically (by loading another 1.5GB model)

## Colab
To run the script on google colab:
1. upload `infer.py`, `reference.wav` to Colab env
2. add and execute code cell `!pip install f5-tts soundfile ruaccent`
3. add and run code cell `!python infer.py`

Requires some GPU to run

## Acknowledgements
Many thanks to [Misha24-10](https://huggingface.co/Misha24-10) for amazing [F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN)
