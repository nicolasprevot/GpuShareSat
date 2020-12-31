pipeline {
    agent any
    triggers {
        pollSCM('* * * * *')
    }
    stages {
        stage("Source") {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: 'refs/heads/master']],
                    extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'sat-infra']],
                    userRemoteConfigs: [[url: '/home/nicolas/prog/sat-infra']]
                ])
            }
        }
        stage('build') {
            steps {
                parallel (
                    simp: {
                        sh 'echo "--solver-type simp-gh --no-release --revision ' + env.BRANCH_NAME + '" | sat-infra/src/schedule.py --short --force --max-wait-time=600 '
                    },
                    gpu: {
                        sh 'echo "--solver-type gpu-gh --no-release --revision ' + env.BRANCH_NAME + '" | sat-infra/src/schedule.py --short --force --max-wait-time=600'
                    }
                )
            }
        }
    }
}
