/*
Multi-Armed Bandit & Agent
Javascript Implementation of https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149


Aidan Nelson, Spring 2019
*/






function pullBandit(bandit) {
    let result = randomGaussian();
    if (result > bandit) {
        return 1;
    } else {
        return -1;
    }
}


class QAgent {
    constructor(_numBandits) {
        this.numBandits = _numBandits;
        this.weights = tf.variable(tf.ones([_numBandits]));

        this.loss = (action, rewardTensor) => tf.log(tf.slice(this.weights, action, [1])).mul(rewardTensor).mul(tf.scalar(-1)).asScalar();


        this.optimizer = tf.train.sgd(0.001);

    }


    async train(_numEpisodes) {
        const e = 0.1;
        let totalReward = [];

        for (let v = 0; v < this.numBandits; v++) {
            totalReward.push(0);
        }

        for (let i = 0; i < _numEpisodes; i++) {


            // choose an action, maybe one at random
            let action;
            if (Math.random() < e) {
                // chose a random action
                action = Math.floor(Math.random() * this.numBandits);
            } else {
                // action = this.model.predict();
                action = tf.tidy(() => {
                    let actionTensor = tf.argMax(this.weights, 0);
                    return actionTensor.dataSync();
                })
            }

            let reward = pullBandit(bandits[action[0]]);

            const rewardTensor = tf.tensor1d([reward]);

            await this.optimizer.minimize(() => this.loss(action, rewardTensor));

            rewardTensor.dispose();

            totalReward[action] += reward;

            if (i % 100 === 0) {
                console.log('Running reward for the ', this.numBandits, ' bandits after ', i, ' trials is: ', totalReward);
            }

        }

        let bestBetIndex = tf.tidy(() => {
            let indexTensor = tf.argMax(this.weights);
            return indexTensor.dataSync();
        });
        
        if (bestBetIndex[0] == bestBanditIndex) {
            console.log("The agent thinks bandit ", bestBetIndex[0] + 1, " is the best bet! And it was right!");
        } else {
            console.log("The agent thinks bandit ", bestBetIndex[0] + 1, " is the best bet! And it was wrong!");
        }

        //         print "The agent thinks bandit " + str(np.argmax(ww) + 1) + " is the most promising...."
        //         if np.argmax(ww) == np.argmax(-np.array(bandits)):
        //             print "...and it was right!"
        // else:
        //         print "...and it was wrong!"
    }
}


let bandits = [0.2, 0, -0.2, -5,-2,-1,1,0,2,3,-1,1,0,2,3,-1,1,0,2,3,-1,1,0,2,3,-1,1,0,2,3];
let bestBanditIndex = 3;


function setup() {
    createCanvas(200, 200);
    // Multi-Armed Bandit

    let qa = new QAgent(bandits.length);
    qa.train(2000);
}

