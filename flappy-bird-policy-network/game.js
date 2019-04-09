/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Implementation based on: http://incompleteideas.net/book/code/pole.c
 */

// import * as tf from '@tensorflow/tfjs';
// import { createStaticVertexBuffer } from '@tensorflow/tfjs-core/dist/kernels/webgl/webgl_util';



// create the p5.js game sketch 
var s = function (sketch) {
  sketch.setup = function () {
    sketch.createCanvas(800, 600);
    sketch.pixelDensity(1);
    sketch.background(0);
  };
};
const sketch = new p5(s, document.getElementById('flappy-bird-canvas-container'));






/**
 * Cart-pole system simulator.
 *
 * In the control-theory sense, there are four state variables in this system:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action:
 *
 *   - leftward or rightward force.
 */
class CartPole {
  /**
   * Constructor of CartPole.
   */
  constructor() {
    // Constants that characterize the system.
    this.gravity = 9.8;
    this.massCart = 1.0;
    this.massPole = 0.5;
    this.totalMass = this.massCart + this.massPole;
    this.cartWidth = 0.2;
    this.cartHeight = 0.1;
    this.length = 0.5;
    this.poleMoment = this.massPole * this.length;
    this.forceMag = 10.0;
    this.tau = 0.02;  // Seconds between state updates.

    // Threshold values, beyond which a simulation will be marked as failed.
    this.xThreshold = 2.4;
    // this.thetaTheshold = 12 / 360 * 2 * Math.PI;
    this.thetaTheshold = 120 / 360 * 2 * Math.PI;

    this.setRandomState();
  }

  /**
   * Set the state of the cart-pole system randomly.
   */
  setRandomState() {
    // The control-theory state variables of the cart-pole system.
    // Cart position, meters.
    this.x = Math.random() - 0.5;
    // Cart velocity.
    this.xDot = (Math.random() - 0.5) * 1;
    // Pole angle, radians.
    this.theta = (Math.random() - 0.5) * 2 * (6 / 360 * 2 * Math.PI);
    // Pole angle velocity.
    this.thetaDot = (Math.random() - 0.5) * 0.5;
  }

  /**
   * Get current state as a tf.Tensor of shape [1, 4].
   */
  getStateTensor() {
    return tf.tensor2d([[this.x, this.xDot, this.theta, this.thetaDot]]);
  }

  /**
   * Update the cart-pole system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  update(action) {
    const force = action > 0 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(this.theta);
    const sinTheta = Math.sin(this.theta);

    const temp =
      (force + this.poleMoment * this.thetaDot * this.thetaDot * sinTheta) /
      this.totalMass;
    const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) /
      (this.length *
        (4 / 3 - this.massPole * cosTheta * cosTheta / this.totalMass));
    const xAcc = temp - this.poleMoment * thetaAcc * cosTheta / this.totalMass;

    // Update the four state variables, using Euler's metohd.
    this.x += this.tau * this.xDot;
    this.xDot += this.tau * xAcc;
    this.theta += this.tau * this.thetaDot;
    this.thetaDot += this.tau * thetaAcc;

    return this.isDone();
  }

  /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `x` (position of the cart) goes out of bound
   * or when `theta` (angle of the pole) goes out of bound.
   *
   * @returns {bool} Whether the simulation is done.
   */
  isDone() {
    return this.x < -this.xThreshold || this.x > this.xThreshold ||
      this.theta < -this.thetaTheshold || this.theta > this.thetaTheshold;
  }
}



/*
Implementation of Flappy Bird as a class for use with tf.js RL algorithm.


References:
tf.js RL 'cart-pole':
https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/cart_pole.js

Dan Shiffman Flappy Bird:
https://github.com/CodingTrain/Flappy-Bird-Clone

Dan Shiffman Neuro-Evolving FLappy Bird: 
https://github.com/CodingTrain/Toy-Neural-Network-JS/blob/master/examples/neuroevolution-flappybird/bird.js

*/

class FlappyBird {

  constructor(canvasWidth, canvasHeight) {
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;
    this.bird = new Bird(this.canvasWidth, this.canvasHeight);
    this.isOver = false;
    this.score = 0;
    this.pipes = [];

    this.pipes.push(new Pipe(this.canvasWidth, this.canvasHeight));
    this.frameCount = 0;
    this.gameoverFrame = this.frameCount - 1;

    this.closest;
  }

  setRandomState() {
    this.isOver = false;
    this.score = 0;
    this.bgX = 0;
    this.pipes = [];
    this.bird = new Bird(this.canvasWidth, this.canvasHeight);
    this.pipes.push(new Pipe(this.canvasWidth, this.canvasHeight));
    this.gameoverFrame = this.frameCount - 1;
  }

  /**
   * Get current state as a tf.Tensor
   */
  getStateTensor() {
    // https://github.com/CodingTrain/Toy-Neural-Network-JS/blob/master/examples/neuroevolution-flappybird/bird.js
    // First find the closest pipe
    let closest = null;
    let record = Infinity;
    for (let i = 0; i < this.pipes.length; i++) {
      let diff = this.pipes[i].x - this.bird.x;
      if (diff > 0 && diff < record) {
        record = diff;
        this.closest = this.pipes[i];
      }
    }
    // Now create the inputs to the neural network
    let inputs = [];
    if (this.closest != null) {
      
      // x position of this.closest pipe
      inputs[0] = mapRanges(this.closest.x, this.bird.x, this.canvasWidth, 0, 1);
      // top of this.closest pipe opening
      inputs[1] = mapRanges(this.closest.top, 0, this.canvasHeight, 0, 1);
      // bottom of this.closest pipe opening
      inputs[2] = mapRanges(this.closest.bottom, 0, this.canvasHeight, 0, 1);
      // bird's y position
      inputs[3] = mapRanges(this.bird.y, 0, this.canvasHeight, 0, 1);
      // bird's y velocity
      inputs[4] = mapRanges(this.bird.velocity, -5, 5, -1, 1);

    } else {
      inputs = [0, 0, 0, 0, 0];
    }
    // console.log('state: ', inputs);
    return tf.tensor2d([inputs]);
  }

  /**
   * Update the FLAPPY BIRD system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  update(action) {
    // console.log('action: ',action);
    let _isDone = false;
    let _reward = false;
    if (action > 0) {
      this.bird.up();
    } else {
      this.bird.down();
    }
    
    // else do nothing

    this.bird.update();

    for (var i = this.pipes.length - 1; i >= 0; i--) {
      this.pipes[i].update();

      if (this.pipes[i].pass(this.bird)) {
        this.score++;
        _reward = true;
      }

      if (this.pipes[i].hits(this.bird)) {
        _isDone = true;
      }

      if (this.pipes[i].offscreen()) {
        this.pipes.splice(i, 1);
      }
    }

    if ((this.frameCount - this.gameoverFrame) % 150 == 0) {
      this.pipes.push(new Pipe(this.canvasWidth, this.canvasHeight));
    }

    this.frameCount++;

    // top of closest pipe opening
    let openingCenterY = mapRanges(this.closest.top + (250/2), 0, this.canvasHeight, 0, 1);
    // bird's y position
    let birdY = mapRanges(this.bird.y, 0, this.canvasHeight, 0, 1);

    // console.log('Closest pipe opening Y: ', openingCenterY);
    // console.log('BirdY: ', birdY);

    let rew = 1 - Math.abs(birdY-openingCenterY);

    // console.log('reward: ', rew);
    // return done state and additional reward for passing pipes
    return [_isDone, rew];
  }

  reset() {
    this.isOver = false;
    this.score = 0;
    this.bgX = 0;
    this.pipes = [];
    this.bird = new Bird(this.canvasWidth, this.canvasHeight);
    this.pipes.push(new Pipe(this.canvasWidth, this.canvasHeight));
    this.gameoverFrame = this.frameCount - 1;
  }

  async show() {
    if (renderDuringTraining) {
      sketch.background(200);
      this.bird.show();
      for (var i = this.pipes.length - 1; i >= 0; i--) {
        this.pipes[i].show();
      }

      // Show inputs:
      let closest = null;
      let record = Infinity;
      for (let i = 0; i < this.pipes.length; i++) {
        let diff = this.pipes[i].x - this.bird.x;
        if (diff > 0 && diff < record) {
          record = diff;
          closest = this.pipes[i];
        }
      }

      if (closest) {
        // console.log('got closest');
        sketch.push();
        sketch.rectMode(sketch.CENTER);
        sketch.fill(200, 100, 200);
        sketch.strokeWeight(2);
        sketch.stroke(0, 0, 255);
        // draw closest pipe as vertical line
        sketch.line(closest.x, 0, closest.x, this.canvasHeight);

        // top of pipe opening as horizontal line
        sketch.line(closest.x, closest.top, closest.x + 80, closest.top);

        // bottom of pipe opening as horizontal line
        sketch.line(closest.x, closest.bottom, closest.x + 80, closest.bottom);

        // console log other parameters
        // console.log("Bird Y Position / Velocity: ", this.bird.y, " / ", this.bird.velocity);
        sketch.pop();
      }
      await tf.nextFrame();  // Unblock UI thread.
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

class Pipe {
  constructor(canvasWidth, canvasHeight) {
    this.canvasHeight = canvasHeight;
    this.spacing = 250;

    this.top = Math.max(Math.floor(Math.random() * Math.floor(3 / 4 * this.canvasHeight)), this.canvasHeight / 6);
    //  random(this.canvasHeight / 6, 3 / 4 * this.canvasHeight);
    this.bottom = this.top + this.spacing;

    this.x = canvasWidth;

    this.w = 80;
    this.speed = 3;

    this.passed = false;
    this.highlight = false;
  }

  hits(bird) {
    let halfBirdHeight = bird.height / 2;
    let halfBirdwidth = bird.width / 2;
    if (bird.y - halfBirdHeight < this.top || bird.y + halfBirdHeight > this.bottom) {
      //if this.w is huge, then we need different collision model
      if (bird.x + halfBirdwidth > this.x && bird.x - halfBirdwidth < this.x + this.w) {
        this.highlight = true;
        this.passed = true;
        return true;
      }
    }
    this.highlight = false;
    return false;
  }

  //this function is used to calculate scores and checks if we've went through the pipes
  pass(bird) {
    if (bird.x > this.x && !this.passed) {
      this.passed = true;
      return true;
    }
    return false;
  }

  // drawHalf() {
  //   let howManyNedeed = 0;
  //   let peakRatio = pipePeakSprite.height / pipePeakSprite.width;
  //   let bodyRatio = pipeBodySprite.height / pipeBodySprite.width;
  //   //this way we calculate, how many tubes we can fit without stretching
  //   howManyNedeed = Math.round(height / (this.w * bodyRatio));
  //   //this <= and start from 1 is just my HACK xD But it's working
  //   for (let i = 0; i < howManyNedeed; ++i) {
  //     let offset = this.w * (i * bodyRatio + peakRatio);
  //     image(pipeBodySprite, -this.w / 2, offset, this.w, this.w * bodyRatio);
  //   }
  //   image(pipePeakSprite, -this.w / 2, 0, this.w, this.w * peakRatio);
  // }

  show() {
    sketch.push();
    sketch.rectMode(sketch.CORNERS);
    sketch.fill(255, 0, 0);
    sketch.noStroke();
    sketch.rect(this.x, 0, this.x + this.w, this.top);
    sketch.rect(this.x, this.bottom, this.x + this.w, this.canvasHeight);
    sketch.pop();
  }

  update() {
    this.x -= this.speed;
  }

  offscreen() {
    return (this.x < -this.w);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

class Bird {
  constructor(canvasWidth, canvasHeight) {

    this.canvasHeight = canvasHeight;
    this.canvasWidth = canvasWidth;

    this.y = this.canvasHeight / 2;
    this.x = 64;


    this.gravity = 0.6;
    this.lift = -5;
    // this.lift = 0;
    this.velocity = 0;

    // this.icon = birdSprite;
    this.width = 64;
    this.height = 64;
  }

  show() {
    // draw the icon CENTERED around the X and Y coords of the bird object
    // image(this.icon, this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
    // console.log('show bird at ', this.x, ' , ', this.y);
    sketch.push();
    sketch.rectMode(sketch.CENTER);
    sketch.fill(0, 0, 255);
    sketch.rect(this.x, this.y, this.width, this.height);
    // sketch.push();
    // sketch.rectMode(sketch.CORNERS);
    // sketch.rect(50,50,100,100);
    // sketch.pop();
    sketch.pop();
  }

  up() {
    this.velocity = this.lift;
    // this.y += 5;
  }

  down() {
    // this.y -= 5;
  }

  update() {
    // uncomment following lines to add gravity back in:
    this.velocity += this.gravity;
    this.y += this.velocity;

    if (this.y >= this.canvasHeight - this.height / 2) {
      this.y = this.canvasHeight - this.height / 2;
      this.velocity = 0;
    }

    if (this.y <= this.height / 2) {
      this.y = this.height / 2;
      this.velocity = 0;
    }

  }
}



function mapRanges(n, start1, stop1, start2, stop2) {
  var newval = (n - start1) / (stop1 - start1) * (stop2 - start2) + start2;
  return newval;
};