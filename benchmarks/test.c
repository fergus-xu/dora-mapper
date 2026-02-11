#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int sum_up_to(int n) {
    int s = 0;
    for (int i = 0; i <= n; i++) {
        s += i;
    }
    return s;
}

int main(void) {
    int x = 3;
    int y = 4;

    int z = add(x, y);
    int total = sum_up_to(z);

    if (total > 10) {
        printf("total = %d\n", total);
    } else {
        printf("small total = %d\n", total);
    }

    return 0;
}
