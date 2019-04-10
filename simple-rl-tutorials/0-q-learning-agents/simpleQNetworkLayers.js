// JS implementation of https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb

class QNetworkLayers {
  constructor(envSize) {
    tf.setBackend('cpu'); // this seems to speed things up a bit
    // create model using TFJS Layers
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: 4,
      useBias: false,
      inputShape: [envSize * envSize],
      kernelInitializer: tf.initializers.randomUniform({
        minval: 0,
        maxval: 0.1
      })
    }));
    this.model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });

    // create a frozenLake environment
    this.envSize = envSize;
    this.env = new FrozenLake(envSize);
    this.env.render();

    // set hyperparameters for training
    this.y = 0.99;
    this.e = 0.2;

    // some variables to store training data:
    this.rList = [];
  }

  async train() {
    let numEpisodes = 100;
    for (let i = 0; i < numEpisodes; i++) {
      // reset current state and environement and rAll
      let currState = this.env.reset();
      let rAll = 0;

      let steps = 0;

      for (let j = 0; j < 100; j++) {
        steps++;

        // Choose an action by greedily (with e chance of random action) from the Q-network...

        // First get all Q values as a tensor
        let t1a = millis();
        const allQ = tf.tidy(() => {
          const inputTensor = this.getStateTensor(currState);
          return this.getAllQ(inputTensor);
        });
        let t1b = millis();

        // then choose the best possible action according to those q values
        let t2a = millis();
        let action = tf.tidy(() => {
          let actionTensor = tf.argMax(allQ, 1);
          return actionTensor.dataSync();
        });
        let t2b = millis();

        // add some randomness! 
        if (Math.random() < this.e) {
          action = [Math.floor(Math.random() * Math.floor(4))];
        }

        // take the action and get the next state and reward from the environment
        let { nextState, reward, done } = this.env.step(action[0]);

        // Obtain maxQ1 (i.e. best action at nextState) and set our target value to be that chosen action:
        let t3a = millis();
        const maxQ1 = tf.tidy(() => {
          const inputTensor = this.getStateTensor(nextState);
          const q1 = this.getAllQ(inputTensor);
          const maxQ1Tensor = tf.argMax(q1, 1);
          return maxQ1Tensor.dataSync();
        });
        let t3b = millis();

        // make a copy of allQ as targetQ
        let t4a = millis();
        let targetQ = tf.tidy(() => {
          return allQ.dataSync();
        });
        let t4b = millis();

        // update targetQ
        targetQ[action[0]] = reward + this.y * maxQ1[0];

        // Train our network using target and predicted Q values
        let t5a = millis();
        const [currStateTensor, targetTensor] = tf.tidy(() => {
          return [this.getStateTensor(currState), tf.tensor2d([targetQ])];
        });
        let t5b = millis();


        let t6a = millis();
        await this.model.fit(currStateTensor, targetTensor, {
          epochs: 1
        });
        let t6b = millis();



        // dispose of remaining tensors:
        let t7a = millis();
        currStateTensor.dispose();
        targetTensor.dispose();
        allQ.dispose();
        let t7b = millis();


        rAll += reward;
        currState = nextState;

        // reduce 'e' chance of taking a random action
        if (done) {
          this.e = 1.0 / ((i / 50) + 10);
          break;
        }
      }

      this.rList.push(rAll);

      if (i % 10 === 0) {
        let numRewards = 0;
        for (let i = 0; i < this.rList.length; i++) { if (this.rList[i] === 1) { numRewards ++; } };
        console.log('Agent successfully reached goal', numRewards, ' out of ', i, ' attempts.');

        if (logBenchmarks) {
          console.log('------------------------ TIME -----------------------');
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
  }

  getAllQ(inputs) {
    return tf.tidy(() => {
      const allQ = this.model.predict(inputs);
      return allQ;
    });
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
        const allQ = this.getAllQ(inputTensor);
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