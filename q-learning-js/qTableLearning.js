
class QAgent {
  constructor(envSize) {
    this.qt = [];

    let numStates = envSize * envSize;
    let numActions = 4;

    // initialize a qtable of zeros
    for (let i = 0; i < numStates; i++) {
      let row = [];
      for (let j = 0; j < numActions; j++) {
        row.push(0);
      }
      this.qt.push(row);
    }

    // initialize the environment:
    this.env = new FrozenLake(envSize, this);
    this.env.render();

    // set hyperparameters (learning rate and y)
    this.lr = 0.8;
    this.y = 0.95;

    // for logging rewards at each training step
    this.rewardList = [];
  }

  train(numEpisodes) {
    for (let i = 0; i < numEpisodes; i++) {
      let currState = this.env.reset();
      let rAll = 0;

      for (let j = 0; j < 100; j++) {


        // the following lines should create a noisey copy of the qtable row at the
        // current state
        let noiseyRow = [];
        // add randomness to each q value at current row:
        for (let m = 0; m < this.qt[currState].length; m++) {
          // noise decreases each episode
          let noise = randomGaussian() * (1.0 / (i + 1.0));
          // console.log("noise: ", noise);
          noiseyRow[m] = noise + this.qt[currState][m];
        }
        // console.log('noisey row: ',noiseyRow);

        // then choose the index of the highest number of that noisey row
        let bestIndex = -1;
        let bestSoFar = -10000;
        for (let m = 0; m < 4; m++) {
          if (noiseyRow[m] > bestSoFar) {
            bestSoFar = noiseyRow[m];
            bestIndex = m;
          }
        }

        // and set our action to that index
        let action = bestIndex;
        // console.log('______________________');
        // console.log('currState: ', currState);
        // console.log('action (LEFT = 0,DOWN = 1,RIGHT = 2,UP = 3): ', action);

        // es6 destructuring syntax
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment
        let { nextState, reward, done } = this.env.step(action);

        // console.log('nextState: ', nextState);
        // console.log('reward: ', reward);
        // console.log('done: ', done);

        let change = this.lr * (reward + this.y * (max(this.qt[nextState])) - this.qt[currState][action]);
        // console.log('change: ',change);
        this.qt[currState][action] = this.qt[currState][action] + change;
        rAll += reward;
        currState = nextState;
        // console.log('updated current state to...', currState);

        if (done) {
          break;
        }
      }
      if (i % 1000 === 0) {
        console.log('_______________');
        console.log('attempt number ', i, ' produced a reward of ', rAll);
        // console.log('qTable: ', this.qt);
      }
      // add rAll to rList
      this.rewardList.push(rAll);
    }
    this.env.reset();
    this.env.render();
  }

  // step through the enviconment once per second
  stepThrough() {
    // console.log("Stepping through...");
    this.env.reset();
    this.env.render();
    this.takeStep();
  }

  takeStep() {
    setTimeout(() => {
      let noiseyRow = [];
      // add randomness to each q value at current row:
      for (let m = 0; m < this.qt[this.env.getCurrentState()].length; m++) {
        // noise decreases each episode
        let noise = 0; // no noise!
        noiseyRow[m] = noise + this.qt[this.env.getCurrentState()][m];
      }
      // console.log('noisey row: ',noiseyRow);

      // then choose the index of the highest number of that noisey row
      let bestIndex = -1;
      let bestSoFar = -10000;
      for (let m = 0; m < 4; m++) {
        if (noiseyRow[m] > bestSoFar) {
          bestSoFar = noiseyRow[m];
          bestIndex = m;
        }
      }
      let action = bestIndex;

      let { nextState, reward, done } = this.env.step(action);
      this.env.render();

      if (!done) {
        this.takeStep();
      }
    }, 500);
  }
}


// let numEpisodes = 10000;
// let env;

// initialize lookup table with all zeros 16x4
// let qt = [[0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0],
// [0, 0, 0, 0]];

// set hyperparameters (learning rate and y)
// let lr = 0.8;
// let y = 0.95;

// let rewardList = [];


// let nowState;
// for debugging
// function keyReleased(){
//   let action = -1;
//   if (keyCode === (LEFT_ARROW)){
//     //move left
//     action = 0;
//   } else if (keyCode === (RIGHT_ARROW)){
//     // move right
//     action = 2;
//   } else if (keyCode === (UP_ARROW)){
//     // move up
//     action = 3;
//   } else if (keyCode === (DOWN_ARROW)){
//     // move down
//     action = 1;
//   }
//   let {nextState, reward, done} = env.step(action);
//
//
//   console.log('____________');
//   console.log('currState: ',nowState );
//   console.log('nextState: ', nextState);
//   console.log('reward: ', reward);
//   console.log('done: ', done);
//
//   nowState = nextState;
//
//   if (done){
//     nowState = env.reset();
//   }
//
// }


// const stepThrough = () => {
//   setTimeout(function () {
//     let noiseyRow = [];
//     // add randomness to each q value at current row:
//     for (let m = 0; m < qt[env.getCurrentState()].length; m++) {
//       // noise decreases each episode
//       let noise = 0; // no noise!
//       noiseyRow[m] = noise + qt[env.getCurrentState()][m];
//     }
//     // console.log('noisey row: ',noiseyRow);

//     // then choose the index of the highest number of that noisey row
//     let bestIndex = -1;
//     let bestSoFar = -10000;
//     for (let m = 0; m < 4; m++) {
//       if (noiseyRow[m] > bestSoFar) {
//         bestSoFar = noiseyRow[m];
//         bestIndex = m;
//       }
//     }
//     let action = bestIndex;

//     let { nextState, reward, done } = env.step(action);
//     env.render();

//     if (!done) {
//       stepThrough();
//     }
//   }, 1000);
// }

// const trainQTable = () => {
//   for (let i = 0; i < numEpisodes; i++) {
//     let currState = env.reset();
//     let rAll = 0;

//     for (let j = 0; j < 100; j++) {


//       // the following lines should create a noisey copy of the qtable row at the
//       // current state
//       let noiseyRow = [];
//       // add randomness to each q value at current row:
//       for (let m = 0; m < qt[currState].length; m++) {
//         // noise decreases each episode
//         let noise = randomGaussian() * (1.0 / (i + 1.0));
//         // console.log("noise: ", noise);
//         noiseyRow[m] = noise + qt[currState][m];
//       }
//       // console.log('noisey row: ',noiseyRow);

//       // then choose the index of the highest number of that noisey row
//       let bestIndex = -1;
//       let bestSoFar = -10000;
//       for (let m = 0; m < 4; m++) {
//         if (noiseyRow[m] > bestSoFar) {
//           bestSoFar = noiseyRow[m];
//           bestIndex = m;
//         }
//       }

//       // and set our action to that index
//       let action = bestIndex;
//       // console.log('______________________');
//       // console.log('currState: ', currState);
//       // console.log('action (LEFT = 0,DOWN = 1,RIGHT = 2,UP = 3): ', action);

//       // es6 destructuring syntax
//       // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment
//       let { nextState, reward, done } = env.step(action);

//       // console.log('nextState: ', nextState);
//       // console.log('reward: ', reward);
//       // console.log('done: ', done);

//       let change = lr * (reward + y * (max(qt[nextState])) - qt[currState][action]);
//       // console.log('change: ',change);
//       qt[currState][action] = qt[currState][action] + change;
//       rAll += reward;
//       currState = nextState;
//       // console.log('updated current state to...', currState);

//       if (done) {
//         break;
//       }
//     }
//     if (i % 1000 === 0) {
//       console.log('_______________');
//       console.log('attempt number ', i, ' produced a reward of ', rAll);
//       console.log('qTable: ', qt);
//     }
//     // add rAll to rList
//     rewardList.push(rAll);
//   }
//   env.reset();
// }
