package com.github.darrmirr.utils;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
public class ArgSortParam implements Comparable<ArgSortParam> {
    private double value;
    @Getter private int index;

    public int compareTo(ArgSortParam obj) {
        double val = this.value - obj.value;
        if (val > 0)
            return 1;
        if (val < 0)
            return -1;
        return 0;
    }
}
