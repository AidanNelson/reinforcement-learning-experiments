// first define a class for the bandit
class Bandit {
    constructor() {
        this.state = 0;
        this.bandits = [[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]];
        this.numBandits = 3;
        this.numActions = 4;
    }

    getBandit() {
        this.state = Math.floor(Math.random() * this.numBandits);
        return this.state;
    }

    pullArm(action) {
        let bandit = this.bandits[this.state][action];
        let r = randomGaussian(0, 1); // normal distribution
        if (r > bandit) {
            return 1;
        } else {
            return -1;
        }
    }
}


// define an agent class
class Agent {
    constructor(_learningRate, _numStates, _numActions) {
        this.lr = _learningRate;
        this.numStates = _numStates;
        this.numActions = _numActions;


        this.weights = tf.variable(tf.ones([this.numStates, this.numActions]));
        // #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        // self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
        // state_in_OH = slim.one_hot_encoding(self.state_in,s_size)
        // output = slim.fully_connected(state_in_OH,a_size,\
        //     biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())
        // self.output = tf.reshape(output,[-1])
        // self.chosen_action = tf.argmax(self.output,0)

        // #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        // #to compute the loss, and use it to update the network.
        // self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        // self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        // self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        // self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        // optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        // self.update = optimizer.minimize(self.loss)

        this.loss = (currState, action, reward) => {
            const rewardTensor = tf.tensor1d([reward]);
            const inputTensor = this.getStateTensor(currState);
            const actionTensor = inputTensor.matMul(this.weights).sigmoid();
            const responsibleWeight = tf.slice(actionTensor.reshape([-1]), action[0], [1]);
            return tf.log(responsibleWeight).mul(rewardTensor).mul(tf.scalar(-1)).asScalar();
        }
        
        this.optimizer = tf.train.sgd(this.lr);


        this.bandit = new Bandit();
        this.totalReward = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        this.e = 0.1;
    }

    async train() {

        for (let i = 0; i < 2000; i++) {
            let currState = this.bandit.getBandit();

            let action;

            if (Math.random() < this.e) {
                action = [Math.floor(Math.random() * this.bandit.numBandits)];
            } else {
                action = tf.tidy(() => {
                    const inputTensor = this.getStateTensor(currState);
                    const actionTensor = inputTensor.matMul(this.weights).sigmoid();
                    return tf.argMax(actionTensor.reshape([-1])).dataSync();
                });
            }

            let reward = this.bandit.pullArm(action[0]);

            
            await this.optimizer.minimize(() => this.loss(currState, action, reward));


            this.totalReward[currState][action[0]] += reward;

            if (i % 100 === 0) {
                console.log('Running reward for the ', this.numStates + 1, ' bandits after ', i, ' trials is: ', this.totalReward);
            }

        }

        // so iterate through it as follows
        for (let i = 0; i < this.numStates; i++) {

            let bestBetIndex = tf.tidy(() => {
                let weightsRow = this.weights.slice([i, 0], [1, 4]).flatten();
                return tf.argMax(weightsRow).dataSync();
            });

            let bestIndex = -1;
            let min = 1000;
            let bandit = this.bandit.bandits[i];

            for (let j = 0; j < this.numActions; j++) {
                if (bandit[j] < min)
                {
                    bestIndex = j;
                    min = bandit[j]
                }
            }

            if (bestBetIndex[0] == bestIndex) {
                console.log("The agent thinks action ", bestBetIndex[0] + 1, " for bandit number ", i + 1, " is the best bet! And it was right!");
            } else {
                console.log("The agent thinks bandit ", bestBetIndex[0] + 1, " for bandit number ", i + 1, " is the best bet! And it was wrong!");
            }

        }
    }

    getStateTensor(s) {
        let ar = [[]];
        for (let i = 0; i < this.numStates; i++) {
            ar[0].push(0);
        }
        ar[0][s] = 1;
        return tf.tensor2d(ar);
    }
}


let myAgent;


function setup() {
    createCanvas(200, 200);
    myAgent = new Agent(0.001, 3, 4);
    myAgent.train(1);
}


