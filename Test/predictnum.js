const brain = require('brain.js');

const net = new brain.recurrent.LSTMTimeStep({
  inputSize: 2,
  hiddenLayers: [10],
  outputSize: 2,
});

// Same test as previous, but combined on a single set
const trainingData = [
  [
    [1, 5],
    [2, 4],
    [3, 3],
    [4, 2],
    [5, 1],
    [5, 2],
    [2, 2],
    [1, 4],
    [2, 5],
    [3, 1],
    [4, 4],
    [5, 3],
    [3, 2],
    [2, 1],
    [5, 5],
    [1, 3],
    [3, 5],
    [5, 4],
    [2, 3],
    [4, 5],
    [1, 1],
    [3, 4],
    [1, 2],
  ],
];

net.train(trainingData, { log: true, errorThresh: 0.09 });

const closeToFiveAndOne = net.run([
  [1, 5],
  [2, 4],
  [3, 3],
  [4, 2],
]);

console.log(closeToFiveAndOne);

// now we're cookin' with gas!
const forecast = net.forecast(
  [
    [1, 5],
    [1, 5],
    [2, 5],
    [3, 5],
    [4, 5],

  ],
  3
);

console.log('next 3 predictions', forecast);
