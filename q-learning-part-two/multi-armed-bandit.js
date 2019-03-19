/*
Multi-Armed Bandit & Agent
Javascript Implementation of https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149


Aidan Nelson, Spring 2019
*/




// Multi-Armed Bandit
let bandits = [0.2,0,-0.2,-5];
let numBandits = len(bandits);

function pullBandit(bandit) {
    let result = randomGaussian();
    if (result > bandit){
        return 1;
    } else {
        return -1;
    }
}




// tf.reset_default_graph()

// #These two lines established the feed-forward part of the network. This does the actual choosing.
// weights = tf.Variable(tf.ones([num_bandits]))
// chosen_action = tf.argmax(weights,0)
class QAgent {
    constructor(_numBandits){
        this.numBandits = _numBandits;
        // this.weights = tf.variable(tf.ones([_numBandits]));
        this.chosenAction = tf.argMax(this.weights,0);

        // this.rewardHolder =  tf.input({shape: [1]});
        // this.actionHolder = tf.input({shape: [1]});
        // this.responsibleWeight = this.weights.slice(actionHolder, [1]);

        //  create model using TFJS Layers
        this.model = tf.sequential();
        this.model.add(
            tf.layers.dense({
            units: [_numBandits],
            useBias: false,
            inputShape: [1],
            kernelInitializer: tf.initializers.ones()
            })
        );

        
        this.model.compile({ 
            optimizer: tf.train.sgd(0.001), 
            loss: tf.losses.logLoss() 
        });
    }


    async train(_numEpisodes) {
        const e = 0.1;
        const totalReward = [];

        for (let v = 0; v < this.numBandits; v++){
            totalReward.push(0);
        }

        for (let i = 0; i < _numEpisodes; i++){
            let action;
            if (Math.random() < e){
                // chose a random action
                action = Math.floor(Math.random() * this.numBandits);
            } else {
                action = this.model.predict();
            }   

            let reward = pullBandit(bandits[action]);
            
            let args = {
                epochs: 1
            };
            await this.model.fit(tf.tensor([action]), tf.tensor([reward]), args);  

        }
        

        // total_episodes = 1000 #Set total number of episodes to train agent on.
        // total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
        // e = 0.1 #Set the chance of taking a random action.

        // init = tf.initialize_all_variables()

        // # Launch the tensorflow graph
        // with tf.Session() as sess:
        //     sess.run(init)
        //     i = 0
        //     while i < total_episodes:
                
        //         #Choose either a random action or one from our network.
        //         if np.random.rand(1) < e:
        //             action = np.random.randint(num_bandits)
            //         else:
            //             action = sess.run(chosen_action)
                    
            //         reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.
                    
            //         #Update the network.
            //         _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
                
        //         #Update our running tally of scores.
        //         total_reward[action] += reward
        //         if i % 50 == 0:
        //             print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
        //         i+=1
        // print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...."
        // if np.argmax(ww) == np.argmax(-np.array(bandits)):
        //     print "...and it was right!"
        // else:
        //     print "...and it was wrong!"
    }
}