# lt25-grpo-debate
GRPO training of LLMs for persuasion in a multi-agent debate setting

Setup:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd 3rdparty/trl && git checkout lt25 && cd ../..
pip install -e 3rdparty/trl/
```

Run the training with:
```
python main.py --config=configs/default.yaml
```

This will set all configurations to the base configurations in `configs/base.yaml`, and override those values with the config that you pass in the `--config` argument. 