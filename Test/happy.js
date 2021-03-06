const brain = require('brain.js');

const trainingData = [
    { input: 'I feel great about the world!', output: 'happy' },
    { input: 'The world is a terrible place!', output: 'sad' },
    { input: 'I am getting hungry!', output: 'sad' },
    { input: 'I/m happy!', output: 'happy' },
];
const config = {
    // Defaults values --> expected validation
    log: (details) => console.log(details),
    iterations: 5000, // the maximum times to iterate the training data --> number greater than 0
    errorThresh: 0.005, // the acceptable error percentage from training data --> number between 0 and 1
    log: true, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 10, // iterations between logging out --> number greater than 0
    learningRate: 0.3, // scales with delta to effect training rate --> number between 0 and 1
    momentum: 0.1, // scales with next layer's change value --> number between 0 and 1
    callback: null, // a periodic call back that can be triggered while training --> null or function
    callbackPeriod: 10, // the number of iterations through the training data between callback calls --> number greater than 0
    timeout: Infinity, // the max number of milliseconds to train for --> number greater than 0
  };


const lstm = new brain.recurrent.LSTM();
const result = lstm.train(trainingData, config);
const json = lstm.toJSON();
    lstm.fromJSON(json);
    lstm.maxPredictionLength = 10000;
console.log('Training result: ', result);

const run1 = lstm.run('I');


console.log(run1);

