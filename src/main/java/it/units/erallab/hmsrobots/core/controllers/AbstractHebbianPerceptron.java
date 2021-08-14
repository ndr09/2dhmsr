package it.units.erallab.hmsrobots.core.controllers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.*;


public abstract class AbstractHebbianPerceptron extends MultiLayerPerceptron {

    @JsonProperty
    protected final MultiLayerPerceptron.ActivationFunction activationFunction;
    @JsonProperty
    protected final double[][][] startingWeights;
    @JsonProperty
    protected final double[][][][] hebbCoef;
    @JsonProperty
    protected final double[][][] eta;
    @JsonProperty
    protected final Set<Integer> disabled;

    @JsonCreator
    public AbstractHebbianPerceptron(
            @JsonProperty("activationFunction") ActivationFunction activationFunction,
            @JsonProperty("weights") double[][][] weights,
            @JsonProperty("hebbCoef") double[][][][] hebbCoef,
            @JsonProperty("neurons") int[] neurons,
            @JsonProperty("eta") double[][][] eta,
            @JsonProperty("disabled") Set<Integer> disabled
    ) {
        super(activationFunction, weights, neurons);
        this.activationFunction = activationFunction;
        this.startingWeights = deepCopy(weights);
        this.hebbCoef = hebbCoef;
        this.eta = eta;
        this.disabled = disabled;
        if (flat(weights, neurons).length != countWeights(neurons)) {
            throw new IllegalArgumentException(String.format(
                    "Wrong number of weights: %d expected, %d found",
                    countWeights(neurons),
                    flat(weights, neurons).length
            ));
        }
        if (flatHebbCoef(hebbCoef, neurons, this.countHebbCoef()).length != this.countHebbCoef(neurons[0], Arrays.stream(this.neurons).skip(1).limit(this.neurons.length - 2).toArray(), neurons[neurons.length - 1])) {
            throw new IllegalArgumentException(String.format(
                    "Wrong number of hebbian coeff: %d   expected, %d found",
                    this.countHebbCoef(neurons[0], Arrays.stream(this.neurons).skip(1).limit(this.neurons.length - 2).toArray(), neurons[neurons.length - 1]),
                    flatHebbCoef(hebbCoef, neurons, this.countHebbCoef()).length
            ));
        }
    }

    /*public AbstractHebbianPerceptron(MultiLayerPerceptron.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, double[] hebbCoef, double eta, Set<Integer> disabled) {
        this(
                activationFunction,
                unflat(weights, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                initEta(eta, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                disabled
        );
    }*/

    public AbstractHebbianPerceptron(HebbianPerceptronFullModelTemp.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] hebbCoef, double eta, Random rnd, Set<Integer> disabled) {
        this(
                activationFunction,
                randW(countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput), rnd),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                initEta(eta, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                disabled
        );
    }

    /*public AbstractHebbianPerceptron(MultiLayerPerceptron.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double eta, Set<Integer> disabled) {
        this(
                activationFunction,
                nOfInput,
                innerNeurons,
                nOfOutput,
                new double[countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput))],
                flatHebbCoef(initHebbCoef(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput)),
                eta,
                disabled);
    }

    public AbstractHebbianPerceptron(HebbianPerceptronFullModelTemp.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, double[] hebbCoef, double[][][] eta, Set<Integer> disabled) {
        this(
                activationFunction,
                unflat(weights, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                eta,
                disabled
        );
    }

    public (HebbianPerceptronFullModelTemp.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] hebbCoef, double[][][] eta, Random rnd, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper) {
        this(
                activationFunction,
                randW(countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput), rnd),
                unflatHebbCoef(hebbCoef, countNeurons(nOfInput, innerNeurons, nOfOutput)),
                countNeurons(nOfInput, innerNeurons, nOfOutput),
                eta,
                disabled,
                mapper
        );
    }

    public HebbianPerceptronFullModelTemp(HebbianPerceptronFullModelTemp.ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[][][] eta, HashSet<Integer> disabled, HashMap<Integer, Integer> mapper) {
        this(activationFunction, nOfInput, innerNeurons, nOfOutput, new double[countWeights(countNeurons(nOfInput, innerNeurons, nOfOutput))], flatHebbCoef(initHebbCoef(countNeurons(nOfInput, innerNeurons, nOfOutput)), countNeurons(nOfInput, innerNeurons, nOfOutput)), eta, disabled, mapper);
    }*/

    public void resetInitWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                System.arraycopy(this.startingWeights[i][j], 0, this.weights[i][j], 0, weights[i][j].length);
            }
        }
    }

    public void hebbianNormalization() {
        for (int l = 1; l < neurons.length; l++) {
            for (int o = 0; o < neurons[l]; o++) {
                double[] v_j = new double[neurons[l - 1]];
                for (int i = 0; i < neurons[l - 1]; i++) {
                    v_j[i] = weights[l - 1][i][o];
                }
                double norm = norm2(v_j);

                for (int i = 0; i < neurons[l - 1]; i++) {
                    weights[l - 1][i][o] /= norm;
                }
            }
        }
    }

    public double[] getStartingWeights() {
        return flat(this.startingWeights, this.neurons);
    }

    @Override
    public double[] getParams() {
        return flatHebbCoef(this.hebbCoef, this.neurons, this.countHebbCoef());
    }

    @Override
    public void setParams(double[] params) {
        double[][][][] tmp = unflatHebbCoef(params, this.neurons);
        for (int i = 0; i < this.neurons.length - 1; i++) {
            for (int j = 0; j < this.neurons[i]; j++) {
                for (int k = 0; k < this.neurons[i + 1]; k++) {
                    System.arraycopy(tmp[i][j][k], 0, this.hebbCoef[i][j][k], 0, 4);
                }
            }
        }
    }

    @Override
    public double[] apply(double[] input) {
        if (input.length != neurons[0]) {
            throw new IllegalArgumentException(String.format("Expected input length is %d: found %d", neurons[0] - 1, input.length));
        }
        double[][] values = new double[neurons.length][];
        values[0] = new double[neurons[0]];
        System.arraycopy(input, 0, values[0], 0, input.length);
        values[0][values[0].length - 1] = 1d; //set the bias
        for (int i = 1; i < neurons.length; i++) {
            values[i] = new double[neurons[i]];
            for (int j = 0; j < neurons[i]; j++) {
                double sum = 0d;
                for (int k = 0; k < neurons[i - 1]; k++) {
                    sum = sum + values[i - 1][k] * weights[i - 1][k][j];
                }
                values[i][j] = activationFunction.apply(sum);
            }
        }
        this.hebbianUpdate(values);
        return values[neurons.length - 1];
    }

    public void hebbianUpdate(double[][] values) {
        for (int l = 1; l < this.hebbCoef.length; l++) {
            for (int o = 0; o < this.neurons[l]; o++) {
                for (int i = 0; i < this.neurons[l - 1]; i++) {
                    if ((l == 1 && !this.disabled.contains(i)) || l > 1) {
                        double dW = this.eta[l - 1][i][o] * (
                                this.hebbCoef[l - 1][i][o][0] * values[l][o] * values[l - 1][i] +   // A*a_i,j*a_j,k
                                        this.hebbCoef[l - 1][i][o][1] * values[l][o] +                 // B*a_i,j
                                        this.hebbCoef[l - 1][i][o][2] * values[l - 1][i] +                 // C*a_j,k
                                        this.hebbCoef[l - 1][i][o][3]                                  // D
                        );
                        this.weights[l - 1][i][o] = this.weights[l - 1][i][o] + dW;
                    }
                }
            }
        }
    }

    public abstract int countHebbCoef(int nOfInput, int[] innerNeurons, int nOfOutput);

    public static double[] flatHebbCoef(double[][][][] hebbCoef, int[] neurons, int size) {
        double[] flatHebbCoef = new double[size];//this.countHebbCoef()];
        int c = 0;
        for (int i = 0; i < hebbCoef.length; i++) {
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    for (int o = 0; o < 4; o++) {
                        flatHebbCoef[c] = hebbCoef[i][j][k][o];
                        c = c + 1;
                    }
                }
            }
        }
        return flatHebbCoef;
    }

    public double[] flatHebbCoef() {
        return flatHebbCoef(this.hebbCoef, this.neurons, this.countHebbCoef());
    }

    public int countHebbCoef() {
        return this.countHebbCoef(this.neurons[0], Arrays.stream(this.neurons).skip(1).limit(this.neurons.length - 2).toArray(), this.neurons[this.neurons.length - 1]);
    }

    public double[][][][] unflatHebbCoef(double[] hebbCoef, int[] neurons) {
        double[][][][] unflatWeights = new double[this.hebbCoef.length][][][];
        int c = 0;
        for (int i = 0; i < this.hebbCoef.length; i++) {
            unflatWeights[i] = new double[neurons[i]][neurons[i + 1]][4];
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    for (int o = 0; o < 4; o++) {
                        unflatWeights[i][j][k][o] = hebbCoef[c];
                        c = c + 1;
                    }
                }
            }
        }
        return unflatWeights;
    }

    public double[][][] initEta(double initEta, int[] neurons) {
        double[][][] unflatEtas = new double[this.hebbCoef.length][][];
        int c = 0;
        for (int i = 0; i < this.hebbCoef.length; i++) {
            unflatEtas[i] = new double[neurons[i]][neurons[i + 1]];
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    unflatEtas[i][j][k] = initEta;
                    c = c + 1;
                }
            }
        }
        return unflatEtas;
    }

    protected static double[][][] deepCopy(double[][][] matrix) {
        double[][][] save = new double[matrix.length][][];
        for (int i = 0; i < matrix.length; i++) {
            save[i] = new double[matrix[i].length][];
            for (int j = 0; j < matrix[i].length; j++)
                save[i][j] = Arrays.copyOf(matrix[i][j], matrix[i][j].length);
        }
        return save;
    }

    protected static double norm2(double[] vector) {
        double norm = 0d;
        for (double v : vector) {
            norm += v * v;
        }
        return Math.sqrt(norm);
    }

    public static double[][][] randW(int nw, int[] neurons, Random rnd) {
        double[] randomWeights = new double[nw];
        for (int i = 0; i < nw; i++) {
            if (rnd != null) {
                randomWeights[i] = (rnd.nextDouble() * 2) - 1;
            } else {
                randomWeights[i] = 0d;
            }
        }
        return unflat(randomWeights, neurons);
    }

    public double[][][][] initHebbCoef(int[] neurons) {
        double[][][][] unflatHebbCoef = new double[this.hebbCoef.length][][][];
        for (int i = 0; i < this.hebbCoef.length; i++) {
            for (int j = 0; j < neurons[i]; j++) {
                for (int k = 0; k < neurons[i + 1]; k++) {
                    unflatHebbCoef[i][j][k] = new double[]{0, 0, 0, 0};
                }
            }
        }
        return unflatHebbCoef;
    }

    public double[] getEta() { return this.flatEta(this.eta); }

    public void setEta(double[] etas) {
        double[][][] newEtas = unflat(etas, neurons);  // TODO: ERROR for output
        for (int l = 0; l < newEtas.length; l++) {
            for (int s = 0; s < newEtas[l].length; s++) {
                System.arraycopy(newEtas[l][s], 0, this.eta[l][s], 0, newEtas[l][s].length);
            }
        }
    }

    public double[] flatEta(double[][][] etaMatrix) {
        double[] etas = new double[this.countHebbCoef() / 4];
        int c = 0;
        for (double[][] matrix : etaMatrix) {
            for (double[] v : matrix) {
                for (double elem : v) {
                    etas[c++] = elem;
                }
            }
        }
        return etas;
    }

    public void setWeights(double[] params) {
        this.setParams(params);
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
        final HebbianPerceptronFullModelTemp other = (HebbianPerceptronFullModelTemp) obj;
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

}
