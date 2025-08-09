#!/usr/bin/env python3
import soundfile as sf
from ruaccent import RUAccent
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
  infer_process,
  load_model,
  load_vocoder,
  preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything

seed_everything(4242)

DEVICE = 'cuda'
WEIGHTS_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors'
VOCAB_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt'
ACCENT_DICT = {
  "реке": "р+еке",
}

vocoder = load_vocoder(device=DEVICE)
ckpt_path = str(cached_path(WEIGHTS_PATH))
vocab_path = str(cached_path(VOCAB_PATH))
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
model_obj = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

accentizer = RUAccent()
accentizer.load(
  omograph_model_size='turbo3.1',
  use_dictionary=True,
  tiny_mode=False,
  custom_dict=ACCENT_DICT
)

def generate(text, out_path, ref):
  ref_file, ref_text = preprocess_ref_audio_text(ref[0], ref[1])
  gen_text = accentizer.process_all(text) + ' '

  wav, sr, _ = infer_process(
    ref_file,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    cross_fade_duration=0.15,
    nfe_step=64,
    speed=1,
    device=DEVICE,
  )

  sf.write(out_path, wav, sr)
  return out_path

ref = (
  'reference.wav',
  'Правительство утвердило финансирование государственной программы «Искусственный интеллект в помощь сельскому жителю».'
)
gen_text = 'Ехал Грека через реку, видит Грека – в реке рак. Сунул Грека руку в реку, - рак за руку Греку цап!'

generate(gen_text, "output.wav", ref)
