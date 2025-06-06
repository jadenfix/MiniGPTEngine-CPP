#include <gtest/gtest.h>
#include <lightgpt/tensor.hpp>
#include <lightgpt/model_loader.hpp>
#include <vector>

using namespace lightgpt;

TEST(TensorFixes, BasicTensorOperations) {
    // Test tensor creation and basic properties
    std::vector<int64_t> shape = {2, 3};
    Tensor t(shape, DType::F32);
    
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.nbytes(), 6 * sizeof(float));
    
    // Test view operation
    auto t_view = t.view({3, 2});
    EXPECT_EQ(t_view.dim(), 2);
    EXPECT_EQ(t_view.shape(), std::vector<int64_t>({3, 2}));
    EXPECT_EQ(t_view.numel(), 6);
}

TEST(TensorFixes, TensorMemoryManagement) {
    // Test memory management
    {
        Tensor t1({2, 2}, DType::F32);
        Tensor t2 = t1;  // Copy constructor
        EXPECT_NE(t1.data<void>(), t2.data<void>());
        
        Tensor t3 = std::move(t1);  // Move constructor
        EXPECT_EQ(t1.data<void>(), nullptr);  // NOLINT(bugprone-use-after-move)
        
        t2 = std::move(t3);  // Move assignment
        EXPECT_EQ(t3.data<void>(), nullptr);  // NOLINT(bugprone-use-after-move)
    }
    // Destructors should be called here without issues
}

// Note: The ModelLoader test is commented out as it requires a model file
/*
TEST(ModelLoaderFixes, BasicModelLoading) {
    // This test requires a model file to be present
    try {
        auto loader = ModelLoader::from_file("path/to/model.gguf");
        auto tensor_names = loader.get_tensor_names();
        EXPECT_FALSE(tensor_names.empty());
    } catch (const std::exception& e) {
        // Skip the test if model file is not available
        GTEST_SKIP() << "Model file not available: " << e.what();
    }
}
*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
