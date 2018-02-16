package nnetworks;

import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;

public class Perceptron{

    private static Random rand;//TODO this is unsafe the way it is conditionally initialized oops

    private Layer[] net;

    private ArrayList<double[]> trainingInputs;
    private ArrayList<double[]> trainingOutputs;

    private double bias;

    public Perceptron(int[] layerSizes, long seed){
        rand = new Random(seed);
        init(layerSizes, seed);
    }

    public Perceptron(int[] layerSizes){
        rand = new Random();
        init(layerSizes, Perceptron.rand.nextLong());
    }

    private void init(int[] layerSizes, long seed){
        net = new Layer[layerSizes.length];
        trainingInputs = new ArrayList<double[]>();
        trainingOutputs = new ArrayList<double[]>();
        bias = 0.0;

        //create layers
        for(int i = 0; i < layerSizes.length; i++){
            net[i] = new Layer(layerSizes[i]);
        }

        //ADD CONNECTIONS:
        for(int layer = net.length - 1; layer > 0; layer--){
        //for each neuron in this layer:
            for(int nthis = 0; nthis < net[layer].neurons.length; nthis++){
                //for each neuron in the last layer
                for(int nlast = 0; nlast < net[layer - 1].neurons.length; nlast++){
                    net[layer].neurons[nthis].addConnection(net[layer - 1].neurons[nlast]);
                }
            }
        }
    }

    public void addTrainingSet(double[] input, double[] expectedOutput){
        trainingInputs.add(input);
        trainingOutputs.add(expectedOutput);
    }

    public void train(int amount){
        for(int i = 0; i < amount; i++){
            train();
        }
    }

    public void train() {
        for(int trainingIndex = 0; trainingIndex < trainingInputs.size(); trainingIndex++){
            double[] outputs = feed(trainingInputs.get(trainingIndex));
            double[] marginError = new double[outputs.length];
            //set the margin of error values: 
            for(int i = 0; i < outputs.length; i++){
                marginError[i] = trainingOutputs.get(trainingIndex)[i] - outputs[i];//figure out why this is all messed up
                System.out.println("\ntraining inputs: " + Arrays.toString(trainingInputs.get(trainingIndex)));
                System.out.println("desired of output #" + i + " = " + trainingOutputs.get(trainingIndex)[i] + ", actual value = " + outputs[i] + ", margin error = " + marginError[i]);
            }

            //for each outputNeuron:
            for(int outputNeuron = 0; outputNeuron < net[net.length - 1].neurons.length; outputNeuron++){
                double error = marginError[outputNeuron];
                //for each layer (except first b/c it has no connections)
                for(int layer = net.length - 1; layer > 0; layer--){
                    //for each neuron 
                    for(int neuron = 0; neuron < net[layer].neurons.length; neuron++){
                        //for each connection 
                        for(Connection connection: net[layer].neurons[neuron].connections){
                            //change the neuron's weight according to the error and slope of the value on the sigmoid func
                            double currVal = net[layer].neurons[neuron].value;  
                            connection.weight += sigmoid(currVal)*(1 - sigmoid(currVal))*(error);
                            System.out.println("adding " + sigmoid(currVal)*(1 - sigmoid(currVal))*(error) 
                                + " to connection (slope = " + sigmoid(currVal)*(1 - sigmoid(currVal)) + ")");
                        }
                    }
                }
            }
        }
    }

    /**
     * takes input weights and feed them through the perceptron, returning the weights of the output nodes
     */
    public double[] feed(double[] input){
        //setting the initial inputs: 
        for(int neuronIndex = 0; neuronIndex < net[0].neurons.length; neuronIndex++){
            net[0].neurons[neuronIndex].value = input[neuronIndex];
        }

        //doing the actual feeding:
        for(int layer = 1; layer < net.length; layer++){
            for(int thisN = 0; thisN < net[layer].neurons.length; thisN++){
                //reset this neuron weight to 0 in case it has been tampered with:
                net[layer].neurons[thisN].value = 0.0;
                for(Connection connection: net[layer].neurons[thisN].connections){
                    net[layer].neurons[thisN].value += (connection.weight * sigmoid(connection.input.value));
                }
                //add bias after summation:
                net[layer].neurons[thisN].value += bias;
            }
        }

        //copying the values from the last layer into return array:
        double[] output = new double[net[net.length - 1].neurons.length];
        for(int i = 0; i < output.length; i++){
            output[i] = net[net.length - 1].neurons[i].value;
        }

        return output;
    }

    private double sigmoid(double input){
        return 1 / (1 + Math.pow(Math.E, -1*input));
    }

    public void setBias(double bias){
        this.bias = bias;
    }

    @Override
    public String toString(){
        StringBuilder b = new StringBuilder();
        for(int i = 0; i < net.length; i++){
            b.append("\n-----LAYER " + i + " ------" + net[i].toString());
        }

        return b.toString() + "\n\n\n";
    }

    private class Layer{ 

        private Neuron[] neurons;

        /**
         * Constructor for Hidden and Output layers
         */
        private Layer(int size){
            neurons = new Neuron[size];
            for(int i = 0; i < size; i++){
                neurons[i] = new Neuron(0);
            }
        }

        /**
         * Constructor for Input Layer
         */
        private Layer(double[] inputWeights){
            neurons = new Neuron[inputWeights.length];
            for(int i = 0; i < inputWeights.length; i++){
                neurons[i] = new Neuron(inputWeights[i]);
            }
        }

        @Override
        public String toString(){
            StringBuilder b = new StringBuilder();
            for(int i = 0; i < neurons.length; i++){
                b.append("\n" + neurons[i].value + ", connection weights: " + neurons[i].connections);
            }

            return b.toString();
        }
    }

    private class Neuron{

        private double value;
        private ArrayList<Connection> connections;

        private Neuron(double initialWeight){
            value = initialWeight;
            connections = new ArrayList<Connection>();
        }

        private void addConnection(Neuron n){
            connections.add(new Connection(n, this));
        }
    }
    
    private class Connection{
        private double weight;
        private Neuron input;
        private Neuron output;

        private Connection(Neuron input, Neuron output){
            this.input = input;
            this.output = output;
            weight = Perceptron.rand.nextDouble();
        }

        @Override
        public String toString(){
            return String.valueOf(weight);
        }
    }
}