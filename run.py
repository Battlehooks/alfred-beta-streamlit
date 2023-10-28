import os
if __name__ == '__main__' :
    # os.system('streamlit run main.py')
    os.system('pip list --format=freeze > requirements.txt')