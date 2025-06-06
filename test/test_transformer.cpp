#include <gtest/gtest.h>
#include <lightgpt/transformer.hpp>

class TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(TransformerTest, BasicTest) {
    // Basic test to verify the test framework is working
    EXPECT_EQ(1, 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
