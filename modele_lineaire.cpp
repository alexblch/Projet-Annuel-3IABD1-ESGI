#include <cstdint>
#include <random>
#include <cmath>
#include <string>
#include <utility>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"
{

    DLLEXPORT int getRandomValue(int max)
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<int> distribution(0, max);

        int randomValue = distribution(generator);
        return randomValue;
    }

    DLLEXPORT int
    set_label_two_output(int index, int fooball_img_list_size, int basket_img_list_size, std::string className)
    {

        std::string class_img;
        if (index >= 0 && index <= fooball_img_list_size - 1)
        {
            class_img = "football";
        }
        else if (index >= fooball_img_list_size && index <= fooball_img_list_size + basket_img_list_size - 1)
        {
            class_img = "basket";
        }
        else
        {
            class_img = "tennis";
        }
        if (class_img == className)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    DLLEXPORT int *set_label_three_output(int fooball_img_list_size, int basket_img_list_size, int tennis_img_list_size)
    {
        int *label_list = new int[fooball_img_list_size + basket_img_list_size + tennis_img_list_size];
        for (int i = 0; i < fooball_img_list_size; i++)
        {
            label_list[i] = 1;
        }
        for (int i = fooball_img_list_size; i < basket_img_list_size + fooball_img_list_size; i++)
        {
            label_list[i] = 2;
        }
        for (int i = basket_img_list_size + fooball_img_list_size; i < tennis_img_list_size + basket_img_list_size + fooball_img_list_size; i++)
        {
            label_list[i] = 3;
        }
        return label_list;
    }

    DLLEXPORT float sigmoid(float output)
    {
        float exp = expf(output);
        return 1 / (1 + (1 / exp));
    }

    DLLEXPORT int get_class_two_output(float output)
    {
        if (sigmoid(output) >= 0.7)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    DLLEXPORT float get_weighted_sum(const int *input, const float *weight, int img_size)
    {
        float weighted_sum = 0;
        for (int i = 0; i < img_size; i++)
        {
            weighted_sum += input[i] * weight[i];
        }
        return weighted_sum;
    }

    DLLEXPORT int get_class_three_output(int *img, float *weight_football, float *weight_basket, float *weight_tennis, int img_size)
    {
        auto *output = new float[3];
        output[0] = sigmoid(get_weighted_sum(img, weight_football, img_size));
        output[1] = sigmoid(get_weighted_sum(img, weight_basket, img_size));
        output[2] = sigmoid(get_weighted_sum(img, weight_tennis, img_size));

        int index_biggest_output = 0;
        for (int i = 1; i < 3; i++)
        {
            if (output[i] > output[index_biggest_output])
            {
                index_biggest_output = i;
            }
        }
        delete[] output;
        return index_biggest_output + 1;
    }

    DLLEXPORT float *initialize_weight(int size)
    {
        auto *weight_list = new float[size];
        for (int i = 0; i < size; i++)
        {
            weight_list[i] = static_cast<float>(getRandomValue(1));
        }
        return weight_list;
    }

    DLLEXPORT void add_bias(int **img_list, int img_list_size, int img_size)
    {
        for (int i = 0; i < img_list_size; i++)
        {
            img_list[i][img_size] = 1;
        }
    }

    DLLEXPORT int get_output_and_set_weight(int *input, float *weight, int weight_list_size, int label)
    {
        float weighted_sum = get_weighted_sum(input, weight, weight_list_size);
        int output = get_class_two_output(weighted_sum);
        for (int i = 0; i < weight_list_size; i++)
        {
            weight[i] = weight[i] + 0.01 * (label - output) * input[i];
        }
        return output;
    }

    DLLEXPORT float **
    train_linear_model(int **img_list, const int *size_class, int img_size, int training_iteration, const std::string &className)
    {
        int fooball_img_list_size = size_class[0];
        int basket_img_list_size = size_class[1];
        int tennis_img_list_size = size_class[2];
        int img_list_size = fooball_img_list_size + basket_img_list_size + tennis_img_list_size;
        add_bias(img_list, img_list_size, img_size);
        auto **weight_and_output = new float *[2];
        float *weight_list = initialize_weight(img_size + 1);
        auto *output_list = new float[training_iteration];
        int random_img_index;
        int label;
        int output;

        for (int i = 0; i < training_iteration; i++)
        {
            random_img_index = getRandomValue(img_list_size - 1);
            label = set_label_two_output(random_img_index, fooball_img_list_size, basket_img_list_size, className);
            output = get_output_and_set_weight(img_list[random_img_index], weight_list, img_size + 1, label);
            if (output == label)
            {
                output_list[i] = 1.0;
            }
            else
            {
                output_list[i] = 0.0;
            }
        }

        weight_and_output[0] = weight_list;
        weight_and_output[1] = output_list;

        return weight_and_output;
    }

    DLLEXPORT int **test_linear_model(int **img_list, const int *size_class, float **weight_list, int img_size)
    {
        int fooball_img_list_size = size_class[0];
        int basket_img_list_size = size_class[1];
        int tennis_img_list_size = size_class[2];
        int img_list_size = fooball_img_list_size + basket_img_list_size + tennis_img_list_size;
        add_bias(img_list, img_list_size, img_size);
        int **label_and_output = new int *[2];
        int *output_list = new int[img_list_size];

        for (int i = 0; i < img_list_size; i++)
        {
            output_list[i] = get_class_three_output(img_list[i], weight_list[0], weight_list[1], weight_list[2], img_size);
        }

        label_and_output[0] = set_label_three_output(fooball_img_list_size, basket_img_list_size, tennis_img_list_size);
        label_and_output[1] = output_list;
        return label_and_output;
    }

    DLLEXPORT float **train_linear_model_football(int **img_list, const int *size_class, int img_size, int training_iteration)
    {
        return train_linear_model(img_list, size_class, img_size, training_iteration, "football");
    }

    DLLEXPORT float **train_linear_model_basket(int **img_list, const int *size_class, int img_size, int training_iteration)
    {
        return train_linear_model(img_list, size_class, img_size, training_iteration, "basket");
    }

    DLLEXPORT float **train_linear_model_tennis(int **img_list, const int *size_class, int img_size, int training_iteration)
    {
        return train_linear_model(img_list, size_class, img_size, training_iteration, "tennis");
    }
}
