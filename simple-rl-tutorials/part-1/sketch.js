/*
A series of Q-Learning experiments built in javascript using tensorflow.js and p5.js
Aidan Nelson, Spring 2019

References:
Arthur Juliani Series on Q Learning: 
  https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
  https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb

Environment based on OpenAI Gym 'frozen lake' environment:
  https://gym.openai.com/envs/FrozenLake-v0/
  https://github.com/openai/gym

Dan Shiffman on tensorflow.js Layers API: 
  https://www.youtube.com/watch?v=F4WWukTWoXY&index=7&list=PLRqwX-V7Uu6YIeVA3dNxbR9PYj4wV31oQ

*/
let qAgent;
let logBenchmarks = false;



function setup(){
    createCanvas(800,800);
    background(200,100,200);

    
    qAgent = new QAgent(6);
    // qAgent = new QNetwork(3);

    let trainButton = createButton('Train Q Agent');
    trainButton.mousePressed(() =>{ qAgent.train()});

    let runButton = createButton('Run Q Agent');
    runButton.mousePressed(() =>{ qAgent.stepThrough()});
    let showButton = createButton('Show Q Table');
    showButton.mousePressed(() =>{ qAgent.env.showTable = true; qAgent.env.render();});
  }
  