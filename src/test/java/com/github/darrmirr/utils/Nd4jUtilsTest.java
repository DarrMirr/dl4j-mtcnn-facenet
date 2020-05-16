package com.github.darrmirr.utils;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

import static org.hamcrest.Matchers.is;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class Nd4jUtilsTest {

    @Test
    public void maximumArg1Double() throws IOException {
        ClassPathResource input = new ClassPathResource("maximum/04_input_arg2_arg1_is_110.ind");
        ClassPathResource output = new ClassPathResource("maximum/04_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.maximum(110D, inputArray);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void maximumArg1Array() throws IOException {
        ClassPathResource input1 = new ClassPathResource("maximum/05_input_arg1.ind");
        ClassPathResource input2 = new ClassPathResource("maximum/05_input_arg2.ind");
        ClassPathResource output = new ClassPathResource("maximum/05_output.ind");
        var inputArray1 = Nd4j.readBinary(input1.getFile());
        var inputArray2 = Nd4j.readBinary(input2.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.maximum(inputArray1, inputArray2);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void minimumArg1Double() throws IOException {
        ClassPathResource input = new ClassPathResource("minimum/03_input_arg2_arg_is_136.ind");
        ClassPathResource output = new ClassPathResource("minimum/03_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.minimum(136D, inputArray);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void minimumArg1Double1() throws IOException {
        ClassPathResource input = new ClassPathResource("minimum/01_input_arg2_arg1_is_73.ind");
        ClassPathResource output = new ClassPathResource("minimum/01_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.minimum(73D, inputArray);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void findFitIndexesLessEquals() throws IOException {
        ClassPathResource input = new ClassPathResource("findFitIndexes/02_input_less_equal_0.5.ind");
        ClassPathResource output = new ClassPathResource("findFitIndexes/02_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.findFitIndexes(inputArray, Conditions.lessThanOrEqual(0.5D));

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void findFitIndexesGreaterEquals() throws IOException {
        ClassPathResource input = new ClassPathResource("findFitIndexes/01_input_greater_equal_0.6.ind");
        ClassPathResource output = new ClassPathResource("findFitIndexes/01_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.findFitIndexes(inputArray, Conditions.greaterThanOrEqual(0.6D));

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void findFitIndexesGreater() throws IOException {
        ClassPathResource input = new ClassPathResource("findFitIndexes/03_input_greater_0.7.ind");
        ClassPathResource output = new ClassPathResource("findFitIndexes/03_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = Nd4jUtils.findFitIndexes(inputArray, Conditions.greaterThan(0.7D));

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }
}
