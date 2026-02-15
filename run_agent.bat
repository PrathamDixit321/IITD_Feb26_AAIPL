@echo off
set HF_HOME=%TEMP%\huggingface
set TORCH_HOME=%TEMP%\torch
echo Setting HF_HOME to %HF_HOME%

echo Running Question Agent...
python question_agent.py --output_file "outputs/questions.json" --num_questions 20 --verbose
pause
