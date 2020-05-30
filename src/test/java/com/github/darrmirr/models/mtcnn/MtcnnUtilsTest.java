package com.github.darrmirr.models.mtcnn;

import com.github.darrmirr.utils.Nd4jUtils;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class MtcnnUtilsTest {
    private MtcnnUtils mtcnnUtils = new MtcnnUtils();

    @Test
    public void imresample() throws IOException {
        ClassPathResource input = new ClassPathResource("imresample/04_input_35.ind");
        ClassPathResource output = new ClassPathResource("imresample/04_output_35.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var scaledArray = Nd4jUtils.imresample(inputArray, 35, 35);

        assertThat(scaledArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void getScoreSortedIndex() throws IOException {
        ClassPathResource input = new ClassPathResource("scoresorted/01_boxes_sort_input.ind");
        ClassPathResource output = new ClassPathResource("scoresorted/01_boxes_sort_output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var scoreSortedArray = mtcnnUtils.getScoreSortedIndex(inputArray);

        assertThat(scoreSortedArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void nms_MethodMinFalse() throws IOException {
        ClassPathResource input = new ClassPathResource("nms/01_input_nms_boxes_0.5_false.ind");
        ClassPathResource output = new ClassPathResource("nms/01_output_nms_boxes_0.5_false.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = mtcnnUtils.nms(inputArray, 0.5, false);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void nms_MethodMinTrue() throws IOException {
        ClassPathResource input = new ClassPathResource("nms/04_input_nms_boxes_0.7_true.ind");
        ClassPathResource output = new ClassPathResource("nms/04_output_nms_boxes_0.7_true.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = mtcnnUtils.nms(inputArray, 0.7, true);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void rerec_01() throws IOException {
        ClassPathResource input = new ClassPathResource("rerec/01-input-160-160.ind");
        ClassPathResource output = new ClassPathResource("rerec/01-output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = mtcnnUtils.rerec(inputArray, 160, 160);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void rerec_03() throws IOException {
        ClassPathResource input = new ClassPathResource("rerec/03-input-160-160.ind");
        ClassPathResource output = new ClassPathResource("rerec/03-output.ind");
        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = mtcnnUtils.rerec(inputArray, 160, 160);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void generateBox() throws IOException {
        ClassPathResource reg = new ClassPathResource("generateBox/01-reg_threshold_0_6_scale_0_6.ind");
        ClassPathResource score = new ClassPathResource("generateBox/01-score.ind");
        ClassPathResource output = new ClassPathResource("generateBox/01-output.ind");
        var regArray = Nd4j.readBinary(reg.getFile());
        var scoreArray = Nd4j.readBinary(score.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());

        var actualArray = mtcnnUtils.generateBox(scoreArray, regArray, 0.6, 0.6);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void getDoubleTest() {
        var array = Nd4j.create(new double[][]{ {1, 2}, {3, 4} });

        for (int i = 0; i < array.columns(); i++) {
            var doubleValue = array.getDouble(1, i);
            System.out.println("double value is " + doubleValue + " for row 1 and column " + i);
        }
    }

    @Test
    public void arrayEquals() {
        var array1 = Nd4j.create(new double[][]{ {1, 2}, {3, 4} });
        var array2 = Nd4j.create(new double[][]{ {1, 2}, {3, 4} });

        assertThat(array1.eq(array2).minNumber(), is(1.0));
    }

    @Test
    public void arrayEqualsDouble() {
        var array1 = Nd4j.create(new double[][]{ {1.01, 2.0}, {3.0, 4.34} });
        var array2 = Nd4j.create(new double[][]{ {1.01, 2.0}, {3.0, 4.34} });

        assertThat(array1.eq(array2).minNumber(), is(1.0));
    }

    @Test
    public void arrayNotEqualsDouble() {
        var array1 = Nd4j.create(new double[][]{ {1.01, 2.0}, {3.0, 4.34} });
        var array2 = Nd4j.create(new double[][]{ {1.01, 2.0}, {3.0, 4.04} });

        assertThat(array1.eq(array2).minNumber(), is(0.0));
    }
}
