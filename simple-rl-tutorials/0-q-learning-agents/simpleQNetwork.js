// JS implementation of https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb

class QNetwork {
  constructor(envSize) {
    tf.setBackend('cpu'); // this seems to speed things up a bit

    // https://js.tensorflow.org/api/latest/#train.sgd
    // this.loss = (pred, label) => pred.sub(label).square().mean();
    this.loss = (Qout, nextQ) => tf.sum(nextQ.sub(Qout).square());
    this.optimizer = tf.train.sgd(0.1);
    this.weights = tf.variable(tf.randomUniform([envSize * envSize, 4], 0, 0.01));
    this.totalTrainingEpisodes = 0;

    // #These lines establish the feed-forward part of the network used to choose actions
    // inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
    // W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    // Qout = tf.matmul(inputs1,W)
    // predict = tf.argmax(Qout,1)

    // #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    // nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
    // loss = tf.reduce_sum(tf.square(nextQ - Qout))
    // trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    // updateModel = trainer.minimize(loss)


    // create a frozenLake environment
    this.envSize = envSize;
    this.env = new FrozenLake(envSize);
    this.env.render();

    // set hyperparameters for training
    this.y = 0.98;
    this.e = 0.2;

    // some variables to store training data:
    this.rList = [];
  }

  async train() {
    let numEpisodes = 10000;
    for (let i = 0; i < numEpisodes; i++) {
      this.totalTrainingEpisodes++;
      await this.trainOneEpisode(i);
    }
  }

  // not sure if necessary to put in separate
  async trainOneEpisode(i) {
    // reset current state and environement and rAll
    let currState = this.env.reset();
    let rAll = 0;

    let t1a, t1b, t2a, t2b, t3a, t3b, t4a, t4b, t5a, t5b, t6a, t6b, t7a, t7b;
    for (let j = 0; j < 100; j++) {
      // Choose an action by greedily (with e chance of random action) from the Q-network...

      // First get all Q values as a tensor
      t1a = millis();
      const allQ = tf.tidy(() => {
        const inputTensor = this.getStateTensor(currState);
        return inputTensor.matMul(this.weights);
      });
      // console.log('__________');
      // console.log('allQ: ',allQ);
      t1b = millis();

      // then choose the best possible action according to those q values
      t2a = millis();
      let action = tf.tidy(() => {
        let actionTensor = tf.argMax(allQ, 1);
        return actionTensor.dataSync();
      });
      t2b = millis();


      // add some randomness! 
      if (Math.random() < this.e) {
        action = [Math.floor(Math.random() * Math.floor(4))];
      }

      // take the action and get the next state and reward from the environment
      let { nextState, reward, done } = this.env.step(action[0]);

      // Obtain maxQ1 (i.e. best action at nextState) and set our target value to be that chosen action:
      t3a = millis();
      const maxQ1 = tf.tidy(() => {
        const inputTensor = this.getStateTensor(nextState);
        const q1 = inputTensor.matMul(this.weights);

        // this should return the max value, not max index
        const maxQ1Tensor = tf.max(q1);
        return maxQ1Tensor.dataSync();
      });
      t3b = millis();

      // make a copy of allQ as targetQ
      t4a = millis();
      const targetQ = tf.tidy(() => {
        return allQ.dataSync().slice();
      });
      t4b = millis();

      // update targetQ
      targetQ[action[0]] = reward + (this.y * maxQ1[0]);

      // Train our network using target and predicted Q values
      t5a = millis();
      const [currStateTensor, targetQTensor] = tf.tidy(() => {
        return [this.getStateTensor(currState), tf.tensor2d([targetQ])];
      });
      t5b = millis();


      t6a = millis();
      // await this.optimizer.minimize(() => this.loss(tf.argMax(currStateTensor.matMul(this.weights), 1), targetQTensor));
      await this.optimizer.minimize(() => this.loss(currStateTensor.matMul(this.weights), targetQTensor));
      t6b = millis();



      // dispose of remaining tensors:
      t7a = millis();
      currStateTensor.dispose();
      targetQTensor.dispose();
      allQ.dispose();
      t7b = millis();


      rAll += reward;
      currState = nextState;

      if (done) {
        // this can be tweaked!
        // reduce 'e' chance of taking a random action
        this.e = 10.0 / ((i / 50.0) + 10.0);
        break;
      }
    }

    this.rList.push(rAll);

    // every now and again, update the console
    if (this.totalTrainingEpisodes % 1000 === 0) {
      let numRewards = 0;
      for (let i = 0; i < this.rList.length; i++) { if (this.rList[i] === 1) { numRewards++; } };
      console.log('-----------------------------------------------------');

      console.log('Agent successfully reached goal', numRewards, ' out of ', this.totalTrainingEpisodes, ' episodes.');
      console.log('Current \'e\' value: ', this.e);

      if (logBenchmarks) {
        console.log('Benchmarks:');
        console.log('Get allQ time:       ', t1b - t1a, ' milliseconds');
        console.log('Get action time:     ', t2b - t2a, ' milliseconds');
        console.log('Get maxQ1 time:      ', t3b - t3a, ' milliseconds');
        console.log('Get targetQ time:    ', t4b - t4a, ' milliseconds');
        console.log('Get tensors time:    ', t5b - t5a, ' milliseconds');
        console.log('Backwards pass time: ', t6b - t6a, ' milliseconds');
        console.log('Disposal time:       ', t7b - t7a, ' milliseconds');
      }
    }
  }

  getStateTensor(s) {
    let ar = [[]];
    for (let i = 0; i < this.envSize * this.envSize; i++) {
      ar[0].push(0);
    }
    ar[0][s] = 1;
    return tf.tensor2d(ar);
  }


  // step through the enviconment once per second
  stepThrough() {
    // console.log("Stepping through...");
    let currState = this.env.reset();
    // this.env.reset();
    this.env.render();
    this.takeStep(currState);
  }

  takeStep(_currState) {
    setTimeout(() => {
      let action = tf.tidy(() => {
        const inputTensor = this.getStateTensor(_currState);
        const allQ = inputTensor.matMul(this.weights);
        let actionTensor = tf.argMax(allQ, 1);
        return actionTensor.dataSync();
      });

      console.log('Agent is at state ', _currState, ' taking action ', action[0]);


      // Get new state and reward from environment
      let { nextState, reward, done } = this.env.step(action[0]);

      this.env.render();

      if (!done) {
        this.takeStep(nextState);
      }
    }, 500);
  }
}