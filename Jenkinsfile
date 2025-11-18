pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/Avi-Kumar-singh/RAG-Chatbot.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Syntax Check') {
            steps {
                sh 'python -m py_compile rag_chatbot.py'
            }
        }

        stage('Run Streamlit App') {
            steps {
                sh '''
                nohup bash -c ". venv/bin/activate && streamlit run rag_chatbot.py --server.port=8501" > streamlit.log 2>&1 &
                '''
            }
        }
    }
}
