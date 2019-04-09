// JS Implementation of https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb

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

  train() {
    let numEpisodes = 1001;
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

        // update the q table at this state
        let change = this.lr * (reward + this.y * (max(this.qt[nextState])) - this.qt[currState][action]);
        this.qt[currState][action] += change;

        rAll += reward;
        currState = nextState;

        if (done) {
          break;
        }
      }
      if (i % 1000 === 0) {
        console.log('_______________');
        // console.log('attempt number ', i, ' produced a reward of ', rAll);
        let numRewards = 0;
        for (let i = 0; i < this.rewardList.length; i++) { if (this.rewardList[i] === 1) { numRewards++; } };
        console.log('Agent successfully reached goal', numRewards, ' out of ', i, ' attempts.');

      }

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

      console.log('Agent is at state ', this.env.getCurrentState(), ' taking action ', action);

      let { nextState, reward, done } = this.env.step(action);
      this.env.render();

      if (!done) {
        this.takeStep();
      }
    }, 500);
  }
}