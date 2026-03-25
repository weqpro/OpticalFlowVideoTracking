#include <gtest/gtest.h>

int add(int a, int b){
    return a+b;
}

TEST(MathTest, AddsPositiveNumbers) {
    // Arrange
    int a = 5;
    int b = 7;

    // Act
    int result = add(a, b);

    // Assert
    EXPECT_EQ(result, 12);
}