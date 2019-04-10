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
        let bandit = this.bandits[this.state][this.action];
        if (randomGaussian() > bandit) {
            return 1;
        } else {
            return -1;
        }
    }
}


// define an agent class

class Agent {
    constructor(_learningRate, _numStates, _numActions) {
        this.model = tf.sequential();
        this.model.add(
            tf.layers.dense({
                units: [_numStates],
                useBias: false,
                activation: 'sigmoid',
                inputShape: [4],
                kernelInitializer: tf.initializers.ones()
            })
        );
        this.model.compile({ 
            optimizer: tf.train.sgd(_learningRate), 
            loss: tf.losses.logLoss 
        });
        this.bandit = new Bandit();
        this.totalReward = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        this.e = 0.01;


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
    }

    async train() {

        for (let i = 0; i < 1; i++) {
            console.log('_________________');
            let currState = this.bandit.getBandit();

            let action;
            if (Math.random() < this.e) {
                action = Math.floor(Math.random() * this.bandit.numBandits);
            } else {
                let stateTensor = this.getStateTensor(currState);
                console.log('stateTensor: ', stateTensor);
                action = this.model.predict(stateTensor);
                let ac = await action.data();
                console.log('action: ', ac);


            }

            let reward = this.bandit.pullArm(action);
            console.log('state: ', currState);
            console.log('reward: ', reward);

        }

        // myAgent = agent(lr=0.001,s_size=cBandit.num_bandits,a_size=cBandit.num_actions) #Load the agent.
        // weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

        // total_episodes = 10000 #Set total number of episodes to train agent on.
        // total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
        // e = 0.1 #Set the chance of taking a random action.

        // init = tf.initialize_all_variables()

        // # Launch the tensorflow graph
        // with tf.Session() as sess:
        //     sess.run(init)
        //     i = 0
        //     while i < total_episodes:
        //         s = cBandit.getBandit() #Get a state from the environment.

        //         #Choose either a random action or one from our network.
        //         if np.random.rand(1) < e:
        //             action = np.random.randint(cBandit.num_actions)
        //         else:
        //             action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]})

        //         reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

        //         #Update the network.
        //         feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
        //         _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

        //         #Update our running tally of scores.
        //         total_reward[s,action] += reward
        //         if i % 500 == 0:
        //             print "Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1))
        //         i+=1
        // for a in range(cBandit.num_bandits):
        //     print "The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising...."
        //     if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        //         print "...and it was right!"
        //     else:
        //         print "...and it was wrong!"
    }

    getStateTensor(s) {
        let oneHotState = [[0, 0, 0, 0]];
        oneHotState[0][s] = 1;
        return tf.tensor2d(oneHotState);
    }
}

let myAgent;
function setup() {
    myAgent = new Agent(0.001, 3, 4);
}

