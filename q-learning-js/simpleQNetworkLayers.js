


class QNetwork {
  constructor(envSize) {
    // create model using TFJS Layers
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: 4,
      useBias: false,
      inputShape: [16],
      kernelInitializer: tf.initializers.randomUniform({
        minval: 0,
        maxval: 0.01
      })
    }));
    this.model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });

    // create a frozenLake environment
    this.env = new FrozenLake(envSize);
    this.env.render();

    // set hyperparameters for training
    this.y = 0.99;
    this.e = 0.1;

    // some variables to store training data:
    this.rList = [];
  }

  async train(numEpisodes) {
    for (let i = 0; i < numEpisodes; i++) {
      // reset current state and environement and rAll
      let currState = this.env.reset();
      let rAll = 0;


      for (let j = 0; j < 100; j++) {
        // Choose an action by greedily (with e chance of random action) from the Q-network
        const allQ = await this.model.predict(this.getStateTensor(currState));
        // console.log(allQ.toString());

        let actionTensor = tf.argMax(allQ, 1);
        let action = await actionTensor.data();
        if (Math.random() < this.e) {
          action = [Math.floor(Math.random() * Math.floor(4))];
        }

        // console.log('currState: ', currState, ', action: ', action[0]);
        // console.log('action: ', action[0]);

        // Get new state and reward from environment
        let { nextState, reward, done } = this.env.step(action[0]);

        // Obtain the Q' values by feeding the new state through our network
        const q1 = await this.model.predict(this.getStateTensor(nextState));
        // console.log('Q1: ', q1.toString());

        // Obtain maxQ' and set our target value for chosen action.
        const maxQ1Tensor = tf.argMax(q1, 1);
        let maxQ1 = await maxQ1Tensor.data();
        // console.log("maxQ1: ", maxQ1[0]);

        let targetQ = await allQ.data();
        // console.log('current q value: ', targetQ);

        targetQ[action] = reward + this.y * maxQ1[0];
        // console.log('desired q value: ', targetQ);

        // Train our network using target and predicted Q values
        await this.model.fit(this.getStateTensor(currState), tf.tensor2d([targetQ]));
        // console.log(response.history.loss[0]);

        // console.log('______________');
        // console.log('currState: ',currState)
        // console.log('action: ', action[0]);
        // console.log('reward: ',reward);
        // console.log('done: ',done);

        rAll += reward;
        currState = nextState;


        allQ.dispose();
        actionTensor.dispose();
        q1.dispose();
        maxQ1Tensor.dispose();

        if (done) {
          this.e = 1.0 / ((i / 50) + 10);
          break;
        }
      }

      this.rList.push(rAll);

      //if (i % 100 === 0){
      // tf.tidy();
      console.log('attempt number ', i, ' produced a reward of ', rAll);
      console.log('tf memory: ', tf.memory());
      //}
    }
  }

  getStateTensor(s) {
    let ar = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
    ar[0][s] = 1;
    return tf.tensor2d(ar);
  }


}


























// let numEpisodes = 5;
// let env;

// // set hyperparameters 
// let y = 0.99;
// let e = 0.1;

// // create model using TFJS Layers
// const model = tf.sequential();
// model.add(tf.layers.dense({
//   units: 4, 
//   useBias: false,
//   inputShape: [16],
//   kernelInitializer: tf.initializers.randomUniform({
//     minval: 0,
//     maxval: 0.01
//   }) 
// }));

// model.compile({optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError'});
// env = new frozenLake();
// trainNetwork();

// // some variables to store training data:
// let rList = [];


// async function trainNetwork() {
//   for (let i = 0; i < numEpisodes; i++) {
//     // reset current state and environement and rAll
//     let currState = this.env.reset();
//     let rAll = 0;


//     for (let j = 0; j < 100; j++) {
//       // Choose an action by greedily (with e chance of random action) from the Q-network
//       const allQ = await model.predict(getStateTensor(currState));
//       // console.log(allQ.toString());

//       let actionTensor = tf.argMax(allQ, 1);
//       let action = await actionTensor.data();
//       if (Math.random() < e) {
//         action = [Math.floor(Math.random() * Math.floor(4))];
//       }

//       // console.log('currState: ', currState, ', action: ', action[0]);
//       // console.log('action: ', action[0]);

//       // Get new state and reward from environment
//       let { nextState, reward, done } = this.env.step(action[0]);

//       // Obtain the Q' values by feeding the new state through our network
//       const q1 = await model.predict(getStateTensor(nextState));
//       // console.log('Q1: ', q1.toString());

//       // Obtain maxQ' and set our target value for chosen action.
//       const maxQ1Tensor = tf.argMax(q1, 1);
//       let maxQ1 = await maxQ1Tensor.data();
//       // console.log("maxQ1: ", maxQ1[0]);

//       let targetQ = await allQ.data();
//       // console.log('current q value: ', targetQ);

//       targetQ[action] = reward + y * maxQ1[0];
//       // console.log('desired q value: ', targetQ);

//       // Train our network using target and predicted Q values
//       await model.fit(getStateTensor(currState), tf.tensor2d([targetQ]));
//       // console.log(response.history.loss[0]);

//       // console.log('______________');
//       // console.log('currState: ',currState)
//       // console.log('action: ', action[0]);
//       // console.log('reward: ',reward);
//       // console.log('done: ',done);

//       rAll += reward;
//       currState = nextState;


//       allQ.dispose();
//       actionTensor.dispose();
//       q1.dispose();
//       maxQ1Tensor.dispose();

//       if (done) {
//         e = 1.0 / ((i / 50) + 10);
//         break;
//       }
//     }

//     rList.push(rAll);

//     //if (i % 100 === 0){
//     console.log('attempt number ', i, ' produced a reward of ', rAll);
//     console.log('tf memory: ', tf.memory());
//     //}
//   }
// }


// function getStateTensor(s) {
//   let ar = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
//   ar[0][s] = 1;
//   return tf.tensor2d(ar);
// }
