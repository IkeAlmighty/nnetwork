import nnetworks.Perceptron;

public class Tester{
    public static void main(String[] args){
        Perceptron net = new Perceptron(
            new int[] { 2, 1 } //size of each layer
        );

        net.addTrainingSet(new double[] { 0, 0 }, new double[] { 0 });
        net.addTrainingSet(new double[] { 1, 1 }, new double[] { 1 });
        net.addTrainingSet(new double[] { 0, 1 }, new double[] { 0 });
        net.addTrainingSet(new double[] { 1, 0 }, new double[] { 0 });

        net.setBias(-0.5);

        net.train(Integer.valueOf(args[0]));

        double[] result = net.feed(new double[] { Integer.valueOf(args[1]), Integer.valueOf(args[2]) });
        for(int i = 0 ; i < result.length; i++){
            System.out.println(result[i]);
        }
    }
}