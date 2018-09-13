@echo off
call activate conda_game 
python .\generate_data.py
if %errorlevel% neq 0 exit /b %errorlevel%
call deactivate
call activate keras_tf
python .\train_and_evaluate.py
call deactivate