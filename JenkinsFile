pipeline {
    agent any

    environment {
        DOCKERHUB_USER = "centronox"
        IMAGE_NAME = "wine-quality-api"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/2022bcs0001-vishnu/LAB_2.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 --version
                pip3 install --upgrade pip
                pip3 install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                python3 scripts/train_2.py
                '''
            }
        }

        stage('Read and Print Metrics') {
            steps {
                sh '''
                echo "===== MODEL METRICS ====="
                cat outputs/results/result.json
                echo ""
                echo "Name: Vishnu Narayanan"
                echo "Roll No: 2022BCS0001"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $DOCKERHUB_USER/$IMAGE_NAME:jenkins .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USERNAME',
                    passwordVariable: 'DOCKER_PASSWORD'
                )]) {
                    sh '''
                    echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
                    docker push $DOCKERHUB_USER/$IMAGE_NAME:jenkins
                    '''
                }
            }
        }
    }
}
