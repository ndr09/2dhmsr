package it.units.erallab.hmsrobots.core.controllers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.snapshots.HMLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Domain;
import it.units.erallab.hmsrobots.util.Parametrized;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class HebbianPerceptronOutputModel implements Serializable, RealFunction, Parametrized, Resettable, Snapshottable {

    public enum ActivationFunction implements Function<Double, Double> {
        RELU(x -> (x < 0) ? 0d : x, Domain.of(0d, Double.POSITIVE_INFINITY)),
        SIGMOID(x -> 1d / (1d + Math.exp(-x)), Domain.of(0d, 1d)),
        SIN(Math::sin, Domain.of(-1d, 1d)),
        TANH(Math::tanh, Domain.of(-1d, 1d));

        private final Function<Double, Double> f;
        private final Domain domain;

        ActivationFunction(Function<Double, Double> f, Domain domain) {
            this.f = f;
            this.domain = domain;
        }

        public Function<Double, Double> getF() {
            return f;
        }

        public Domain getDomain() {
            return domain;
        }

        public Double apply(Double x) {
            return f.apply(x);
        }
    }

    @JsonProperty
    private final ActivationFunction activationFunction;
    @JsonProperty
    private final double[][][] weights;
    @JsonProperty
    private final double[][][] startingWeights;
    @JsonProperty
    double[][][] hebbCoef;
    @JsonProperty
    private final int[] neurons;
    @JsonProperty
    private final double[][] eta;
    @JsonProperty
    private final HashSet<Integer> disabled;
    @JsonProperty
    private final HashMap<Integer, Integer> mapper;
    @JsonProperty
    private final double[] normalization;

    private double[][] activationsValues;

    @JsonCreator
    public HebbianPerceptronOutputModel(
            @JsonProperty("activationFunction") ActivationFunction activationFunction,
            @JsonProperty("weights") double[][][] weights,
            @JsonProperty("hebbCoef") double[][][] hebbCoef,
            @JsonProperty("neurons") int[] neurons,
            @JsonProperty("eta") double[][] eta,
            @JsonProperty("disabled") HashSet<Integer> disabled,
            @JsonProperty("mapper") HashMap<Integer, Integer> mapper,
            @JsonProperty("normalization") double[] normalization
    ) {
        this.activationFunction = activationFunction;
        this.weights = weights;
        this.startingWeights = deepCopy(weights);
        this.neurons = neurons;
        this.hebbCoef = hebbCoef;
        this.eta = eta;
        /*String et ="";
        for (double d:flatEta(this.eta,neurons)){
            et += d+" ";
        }
        System.out.println("eta "+flatEta(this.eta, neurons).length+" "+et);*/
        this.disabled = disabled;
        this.normalization =normalization;
        if (flat(weights, neurons).length != countWeights(neurons)) {
            throw new IllegalArgumentException(String.format(
                    "Wrong number of weights: %d expected, %d found",
                    countWeights(neurons),
                    flat(weights, neurons).length
            ));
        }
        this.mapper = mapper;
        //System.out.println(neurons[neurons.length-1]);
        //System.out.println(neurons.length-1);
        if (flatHebbCoef(hebbCoef, neurons).length != 4 * ((Arrays.stream(neurons).sum()) - neurons[0])) {
            throw new IllegalArgumentException(String.format(
                    "Wrong number of hebbian coeff: %d   expected, %d found",
                    4 * neurons[neurons.length - 1],
                    flatHebbCoef(hebbCoef, neurons).length
            ));
        }
    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, double[] hebbCoef, double eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper, double[] normalization) {
        this(
                activationFunction,
                unflat(weights, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                initEta(eta,countNeurons(nOfInput, innerNeurons, nOfOutput)),
                disabled,
                mapper,
                normalization
        );
    }
    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, double[] hebbCoef, double eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper) {
        this(
                activationFunction,
                unflat(weights, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                initEta(eta,countNeurons(nOfInput, innerNeurons, nOfOutput)),
                disabled,
                mapper,
                null
        );
    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput,
                                                    double[] hebbCoef, double eta, Random rnd, HashSet<Integer> disabled,
                                                    HashMap<Integer, Integer> mapper, double[] normalization) {

        this(
                activationFunction,
                randW(countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput), rnd),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                initEta(eta,countNeurons(nOfInput, innerNeurons, nOfOutput)),
                disabled,
                mapper, normalization
        );
    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper, double[] normalization) {
        this(activationFunction, nOfInput, innerNeurons, nOfOutput,
                new double[countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput))],
                flatHebbCoef(initHebbCoef(countNeurons(nOfInput, innerNeurons, nOfOutput)),
                        countNeurons(nOfInput, innerNeurons, nOfOutput)), initEta(eta,countNeurons(nOfInput, innerNeurons, nOfOutput)) , disabled, mapper, normalization);

    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, double[] hebbCoef, double[][] eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper, double[] normalization) {
        this(
                activationFunction,
                unflat(weights, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                eta,
                disabled,
                mapper,
                normalization
        );
    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] hebbCoef, double[][] eta, Random rnd, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper, double[] normalization) {

        this(
                activationFunction,
                randW(countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput), rnd),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                eta,
                disabled,
                mapper,
                normalization
        );
    }

    public HebbianPerceptronOutputModel(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[][] eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper, double[] normalization) {
        this(activationFunction, nOfInput, innerNeurons, nOfOutput,
                new double[countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput))],
                flatHebbCoef(initHebbCoef(countNeurons(nOfInput, innerNeurons, nOfOutput)),
                        countNeurons(nOfInput, innerNeurons, nOfOutput)), eta, disabled, mapper, normalization);

    }
    public static double[][][] unflat(double[] flatWeights, int[] neurons) {
        double[][][] unflatWeights = new double[neurons.length - 1][][];
        int c = 0;
        for (int i = 0; i < neurons.length - 1; i++) {
            unflatWeights[i] = new double[neurons[i]][neurons[i + 1]];
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    unflatWeights[i][j][k] = flatWeights[c];
                    c = c + 1;
                }
            }
        }
        return unflatWeights;
    }

    public static double[][][] unflatHebbCoef(double[] hebbCoef, int[] neurons) {

        double[][][] unflatWeights = new double[neurons.length-1][][];
        int c = 0;
        //System.out.println("^^^^ "+neurons.length);
        for (int l = 1; l<neurons.length; l++){
            unflatWeights[l-1] = new double[neurons[l]][];
        for (int i = 0; i < neurons[l]; i++) {
            unflatWeights[l-1][i] = new double[4];
            for (int w = 0; w < 4; w++) {
                unflatWeights[l-1][i][w] = hebbCoef[c];
                c = c + 1;
            }
        }

        }
        return unflatWeights;
    }

    public static double[] flat(double[][][] unflatWeights, int[] neurons) {
        int n = 0;
        for (int i = 0; i < neurons.length - 1; i++) {
            n = n + neurons[i] * neurons[i + 1];
        }
        double[] flatWeights = new double[n];
        int c = 0;
        for (int i = 0; i < neurons.length - 1; i++) {
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    flatWeights[c] = unflatWeights[i][j][k];
                    c = c + 1;
                }
            }
        }

        return flatWeights;
    }

    public static double[] flatHebbCoef(double[][][] hebbCoef, int[] neurons) {
        int n = 4 * ((Arrays.stream(neurons).sum()) - neurons[0]); //first layer does not have hebb coeff
        double[] flatHebbCoef = new double[n];
        int c = 0;
        for (int l = 0; l < hebbCoef.length; l++) {
            for (int i = 0; i < neurons[l+1]; i++) {
                for (int w = 0; w < 4; w++) {
                    flatHebbCoef[c] = hebbCoef[l][i][w];
                    //System.out.println("@@ "+flatHebbCoef[c]);
                    c = c + 1;
                }
            }
        }
        //System.out.println("@@## "+c);
        return flatHebbCoef;
    }

    public static double[][][] initHebbCoef(int[] neurons) {
        int n = 4 * ((Arrays.stream(neurons).sum()) - neurons[0]); //first layer does not have hebb coeff
        double[][][] unflatHebbCoef = new double[n][][];
        int c = 0;
        for (int l= 1;l<neurons.length;l++) {
            unflatHebbCoef[l] = new double[neurons[l]][];
            for (int i = 0; i < n; i++) {
                unflatHebbCoef[l][i] = new double[]{0, 0, 0, 0};
            }
        }
        return unflatHebbCoef;
    }

    public static int[] countNeurons(int nOfInput, int[] innerNeurons, int nOfOutput) {
        final int[] neurons;
        neurons = new int[2 + innerNeurons.length];
        System.arraycopy(innerNeurons, 0, neurons, 1, innerNeurons.length);
        neurons[0] = nOfInput;
        neurons[neurons.length - 1] = nOfOutput;
        return neurons;
    }

    public static int countWeights(int[] neurons) {
        int largestLayerSize = Arrays.stream(neurons).max().orElse(0);
        double[][][] fakeWeights = new double[neurons.length][][];
        double[][] fakeLayerWeights = new double[largestLayerSize][];
        Arrays.fill(fakeLayerWeights, new double[largestLayerSize]);
        Arrays.fill(fakeWeights, fakeLayerWeights);
        return flat(fakeWeights, neurons).length;
    }

    public static int countWeights(int nOfInput, int[] innerNeurons, int nOfOutput) {
        return countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput));
    }

    public static int countHebbCoef(int nOfInput, int[] innerNeurons, int nOfOutput) {
        return 4* Arrays.stream(countNeurons(nOfInput, innerNeurons, nOfOutput)).sum()-nOfInput;
    }

    private double norm2(double[] vector) {
        double norm = 0d;
        for (int i = 0; i < vector.length; i++) {
            norm += vector[i] * vector[i];
        }
        return Math.sqrt(norm);

    }

    public void hebbianUpdate(double[][] values) {
        double[] wei1 = flat(this.weights, neurons);
        for (int l = 1; l < neurons.length; l++) {
            for (int o = 0; o < neurons[l]; o++) {
                for (int i = 0; i < neurons[l - 1]; i++) {
                    if ((l == 1 && !disabled.contains(i)) || l > 1) {
                        //System.out.println(o);
                        double dW = eta[l-1][o] * (
                                hebbCoef[l-1][o][0] * values[l][o] * values[l - 1][i] +   // A*a_i,j*a_j,k
                                        hebbCoef[l-1][o][1] * values[l][o] +                 // B*a_i,j
                                        hebbCoef[l-1][o][2] * values[l - 1][i] +                 // C*a_j,k
                                        hebbCoef[l-1][o][3]                                  // D
                        );
                        weights[l - 1][i][o] = weights[l - 1][i][o] + dW;
                    }
                }
            }
        }
    }

    @Override
    public void reset() {
        this.resetInitWeights();
    }

    @Override
    public Snapshot getSnapshot() {
        return new Snapshot(new HMLPState(this.activationsValues, this.weights, this.activationFunction.domain), this.getClass());
    }

    public void hebbianNormalization() {
        for (int l = 1; l < neurons.length; l++) {
            for (int o = 0; o < neurons[l]; o++) {
                double[] v_j = new double[neurons[l - 1]];
                for (int i = 0; i < neurons[l - 1]; i++) {
                    v_j[i] = weights[l - 1][i][o];
                }
                double[] norm;

                if (this.normalization[0] == this.normalization[1]) {
                    norm = norm(v_j);
                } else {
                    norm = bound(v_j);
                }

                for (int i = 0; i < neurons[l - 1]; i++) {

                    weights[l - 1][i][o] = norm[i];

                    //weights[l - 1][i][o] *= 100;
                }
            }
        }

    }


    private double[] norm(double[] vector) {
        double mmin = Arrays.stream(vector).min().getAsDouble();
        double mmax = Arrays.stream(vector).max().getAsDouble();
        if (mmin != mmax) {
            for (int i = 0; i < vector.length; i++) {
                Double tmp = 2 * ((vector[i] - mmin) / (mmax - mmin)) - 1;
                if (tmp.isNaN()) {
                    System.out.println(Arrays.toString(vector));
                    System.out.println(vector[i]);
                    System.out.println(mmin);
                    System.out.println(mmax);
                    throw new IllegalArgumentException(String.format("nan weights"));
                }
                vector[i] = tmp;
            }
        }

        return vector;
    }

    private double[] bound(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] < this.normalization[0]) {
                vector[i] = this.normalization[0];
            }
            if (vector[i] > this.normalization[1]) {
                vector[i] = this.normalization[1];
            }
        }
        return vector;
    }

    public void resetInitWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    this.weights[i][j][k] = this.startingWeights[i][j][k];
                }
            }
        }
    }

    private double[][][] deepCopy(double[][][] matrix) {
        double[][][] save = new double[matrix.length][][];
        for (int i = 0; i < matrix.length; i++) {
            save[i] = new double[matrix[i].length][];
            for (int j = 0; j < matrix[i].length; j++)
                save[i][j] = Arrays.copyOf(matrix[i][j], matrix[i][j].length);
        }
        return save;
    }

    @Override
    public double[] apply(double[] input) {
        if (input.length != neurons[0]) {
            throw new IllegalArgumentException(String.format("Expected input length is %d: found %d", neurons[0] - 1, input.length));
        }
        activationsValues = new double[neurons.length][];
        activationsValues[0] = new double[neurons[0]];
        System.arraycopy(input, 0, activationsValues[0], 0, input.length);
        activationsValues[0][activationsValues[0].length - 1] = 1d; //set the bias
        for (int i = 1; i < neurons.length; i++) {
            activationsValues[i] = new double[neurons[i]];
            for (int j = 0; j < neurons[i]; j++) {
                double sum = 0d;
                for (int k = 0; k < neurons[i - 1]; k++) {
                    sum = sum + activationsValues[i - 1][k] * weights[i - 1][k][j];
                }
                activationsValues[i][j] = activationFunction.f.apply(sum);
            }
        }

        hebbianUpdate(activationsValues);
        double[] pre = this.getWeights();
        if (!(this.normalization == null)) {
            hebbianNormalization();
        }
        double[] post = this.getWeights();
        //System.out.println(Arrays.toString(IntStream.range(0, post.length).mapToDouble(i ->pre[i]-post[i]).toArray()));

        return activationsValues[neurons.length - 1];
    }

    public double[][][] getWeightsMatrix() {
        return weights;
    }

    public int[] getNeurons() {
        return neurons;
    }


    public double[] getWeights() {
        return flat(weights, neurons);
    }

    public double[] getStartingWeights() {
        return flat(startingWeights, neurons);
    }

    @Override
    public double[] getParams() {

        return flatHebbCoef(hebbCoef, neurons);
    }

    public double[] flatEta(double[][] etaMatrix, int[] neurons){
        double[] etas = new double[((Arrays.stream(neurons).sum()) - neurons[0])];
        int c =0;
        for (int l = 0; l<etaMatrix.length; l++){
            for (int o = 0; o<etaMatrix[l].length;o++){
                etas[c] = etaMatrix[l][o];
                c++;
            }
        }
        return etas;
    }

    public double[][] unflatEta(double[] etasArr, int[] neurons){
        double[][] etas = new double[neurons.length-1][];
        int c=0;
        for(int l =1; l<neurons.length;l++){
            etas[l-1] = new double[neurons[l]];
            for (int o =0; o<etas[l-1].length;o++){
                etas[l-1][o] = etasArr[c];
                c++;
            }
        }
        return etas;
    }

    public void setWeights(double[] params) {
        double[][][] newWeights = MultiLayerPerceptron.unflat(params, neurons);
        for (int l = 0; l < newWeights.length; l++) {
            for (int s = 0; s < newWeights[l].length; s++) {
                for (int d = 0; d < newWeights[l][s].length; d++) {
                    weights[l][s][d] = newWeights[l][s][d];
                }
            }
        }
    }

    public double[] getEta(){return flatEta(eta, neurons);}

    public void setEta(double[] etas){
        double[][] newEtas = unflatEta(etas, neurons);
        for (int l = 0; l < newEtas.length; l++) {
            for (int s = 0; s < newEtas[l].length; s++) {
                    eta[l][s] = newEtas[l][s];
            }
        }
    }

    public static double[][] initEta(double initEta, int[] neurons){
        double[][] unflatEtas = new double[neurons.length - 1][];
        int c = 0;
        for (int l = 1; l < neurons.length; l++) {
            unflatEtas[l-1] = new double[neurons[l]];
            for (int j = 0; j < neurons[l]; j++) {
                    unflatEtas[l-1][j] = initEta;
                    c = c + 1;
                    //System.out.println(c);

            }
        }
        return unflatEtas;
    }

    @Override
    public void setParams(double[] params) {
        double[][][] tmp = unflatHebbCoef(params, neurons);
        for (int l=1; l< neurons.length; l++) {
            for (int i = 0; i < neurons[l]; i++) {
                System.out.println(l+"    "+neurons[l]);
                for (int j = 0; j < 4; j++) {
                    hebbCoef[l][i][j] = tmp[l][i][j];
                }
            }
        }
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 67 * hash + Objects.hashCode(this.activationFunction);
        hash = 67 * hash + Arrays.deepHashCode(this.weights);
        hash = 67 * hash + Arrays.hashCode(this.neurons);
        hash = 67 * hash + Arrays.hashCode(this.hebbCoef);
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final HebbianPerceptronOutputModel other = (HebbianPerceptronOutputModel) obj;
        if (this.activationFunction != other.activationFunction) {
            return false;
        }
        if (!Arrays.deepEquals(this.weights, other.weights)) {
            return false;
        }
        if (!Arrays.deepEquals(this.hebbCoef, other.hebbCoef)) {
            return false;
        }
        return Arrays.equals(this.neurons, other.neurons);
    }

    @Override
    public String toString() {
        return "MLP." + activationFunction.toString().toLowerCase() + "[" +
                Arrays.stream(neurons).mapToObj(Integer::toString).collect(Collectors.joining(","))
                + "]";
    }

    public static double[][][] randW(int nw, int[] neurons, Random rnd) {
        double[] randomWeights = new double[nw];
        for (int i = 0; i < nw; i++) {
            if (rnd != null) {
                //System.out.println("not null");
                randomWeights[i] = (rnd.nextDouble() * 2) - 1;
            } else {
                //System.out.println("null");
                randomWeights[i] = 0d;
            }
        }
        return unflat(randomWeights, neurons);
    }

    @Override
    public int getInputDimension() {
        return neurons[0];
    }

    @Override
    public int getOutputDimension() {
        return neurons[neurons.length - 1];
    }

}
