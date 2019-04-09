// JS Implementation of https://gym.openai.com/envs/FrozenLake-v0/
class FrozenLake {
  constructor(size, qAgent) {
    this.size = size;
    this.map = [];

    this.qAgent = qAgent;
    this.showTable = false;
    
    for (let i = 0; i < size; i++){
      let row = [];
      for (let j = 0; j < size; j++){
        let next;
        if (Math.random() < 0.25){
          next = "H";
        } else {
          next = "F";
        }
        
        if (i == 0 && j == 0){
          next = "S";
        }
        if (i == size-1 && j == size-1){
          next = "G";
        }

        row.push(next);
      }
      this.map.push(row);
    }
    // this.map =
    //   [["S", "F", "F", "F"],
    //   ["F", "H", "F", "H"],
    //   ["F", "F", "F", "H"],
    //   ["H", "F", "F", "G"]];

    // agent begins at top left position
    this.pos = { x: 0, y: 0 };

  }

  reset() {
    //Reset the environment's state. Returns observation.
    this.pos.x = 0; this.pos.y = 0;
    return this.getCurrentState();
  }

  step(action) {
    // Step the environment by one timestep. Returns observation, reward, done

    // move the player
    if (action == 0) { // left
      this.pos.x = Math.max(this.pos.x - 1, 0);
    } else if (action == 1) { // down
      this.pos.y = Math.min(this.pos.y + 1, this.size-1);
    } else if (action == 2) { // right
      this.pos.x = Math.min(this.pos.x + 1, this.size-1);
    } else if (action == 3) { // up
      this.pos.y = Math.max(this.pos.y - 1, 0);
    }

    let reward = 0;
    let done = false;

    if (this.map[this.pos.y][this.pos.x] == "G") {
      reward = 1;
      done = true;
    }
    if (this.map[this.pos.y][this.pos.x] == "H") {
      done = true;
    }

    return {
      nextState: this.getCurrentState(),
      reward: reward,
      done: done
    }
  }

  render() {
    // reset background
    background(200, 100, 200);

    let side = width > height ? height : width;

    // render one frame of the environment.
    let buff = side / 6;
    let spaceSize = (side - buff) / this.size;



    for (let x = 0; x < this.map[0].length; x++) {
      for (let y = 0; y < this.map.length; y++) {

        
        let em = "";
        switch (this.map[y][x]) {
          case "S":
            em = "🔸";
            break;
          case "F":
            em = "🔸";
            break;
          case "H":
            em = "☠";
            break;
          case "G":
            em = "🎉";
            break;
        }

        textSize(64);
        textAlign(CENTER);
        text(em, buff + x * spaceSize, (buff + y * spaceSize));

        
        if (this.showTable){
          let stateIndex = (y * this.map[0].length) + x;
          let actionArr = this.qAgent.qt[stateIndex];
          let actionIndex = -1;
          let maxSoFar = -1000;
        
          for (let p = 0; p < actionArr.length; p++){

            if (actionArr[p] > maxSoFar) {
              actionIndex = p;
              maxSoFar = actionArr[p];
            }
          }
          // console.log('action at x = ',x, ' / y = ', y, ' --> ',actionIndex);
          let arrow = '';
          switch (actionIndex) {
            case 0:
              arrow = "⬅";
              break;
            case 1:
              arrow = "⬇";
              break;
            case 2:
              arrow = "➡";
              break;
            case 3:
              arrow = "⬆";
              break;
          }
          textSize(64);
          stroke(0);
          fill(200,200,200,200);
          textAlign(CENTER);
          text(arrow, buff + x * spaceSize, (buff + y * spaceSize));
        }

        
      }
    }

    background(255,255,255,50);
    let em = "";
    // render player position
    switch (this.map[this.pos.y][this.pos.x]) {
      case "S":
        em = '😐';
        break;
      case "F":
        em = '😐';
        break;
      case "H":
        em = '😱';
        break;
      case "G":
        em = '🤩';
        break;
    }

    text(em, this.pos.x * spaceSize + buff, this.pos.y * spaceSize + buff);
  }

  getCurrentState() {
    // return a number from 0-15 for current state of player
    return this.pos.y * this.size + this.pos.x;
  }
}
