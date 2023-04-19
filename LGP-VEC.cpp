#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>

#include "BS_thread_pool.hpp"
// #include "matplotlibcpp.h"

#include "primitives.hpp"
#include "data_parser_v2.hpp"

#define BENCHMARK 7 // Benchmark function to run
#define THREADS   8 // Number of threads to use

using MATRIX    = Eigen::Tensor<double, 2>;
using REGISTERS = MATRIX;
using DATA      = Eigen::Tensor<double, 3>;
using iptr      = MATRIX (*)(const MATRIX& a, const MATRIX& b); // primitive function
std::mutex      g_losers_mutex;   // Mutex for guarding the global losers of tournaments
std::mutex      g_children_mutex; // Mutex for guaring the global children for new generations
std::mutex      g_fitness_mutex;  // Mutex for guarding the global best fitness var
BS::thread_pool concurrent_tournaments(THREADS);

#define ARITY     2
#define CONSTS    20  // Number of constant registers
#define REG_F     3   // Scaling factor for the number of registers. Experiment with different values
#define CONST_MIN 0.0 // Min values of constants (remember to add decimal if needed)
#define CONST_MAX 2.0

#define GN              250             // Number of genes in the chromosome (i.e. length of the program)
#define POP_SIZE        500             // Population size
#define TOURNAMENT_SIZE 0.05 * POP_SIZE // Experiment with different values
#define CR              0.93            // Crossover rate. Exepriment diff values
#define MAX_GEN         500             // Maximum number of generations
#define F_MACRO         0.9                // Macro mutation rate (exploration)
#define F_MACRO_DEL     0.35            // Macro mutation rate for deletion
#define F_MACRO_INS     0.65            // Macro mutation rate for insertion
#define F_MICRO         0.5             // Micro mutation rate (exploitation)

// For statistical analysis
#define RUNS 30
std::vector<double> best_individuals; // Used for the graph
std::vector<double> testing_fitness;

auto                   best_fitness   = MAXFLOAT;
int                    registers      = 1; // Input arguments (Each argument is a vector/MATRIX).
int                    training_cases = 1;
int                    testing_cases  = 1;
int                    max_dim        = 1; // Training input dimension
int                    max_dim_out    = 1; // Output dimension
DATA                   training_data;
DATA                   testing_data;
MATRIX                 training_output; // Matrix of size training_cases*max_dim. if scalar -> const matrix
MATRIX                 testing_output;
MATRIX                 output; // Global output matrix. Both training and testing for better readability
std::vector<REGISTERS> R;      // Global training registers
std::vector<REGISTERS> S;      // Global testing registers

enum class TYPE { TRAINING, TESTING };

/// @brief Gives a random instruction (function pointer) for the initialisation/mutation of the chromosome
iptr rand_instruction()
{
    auto rand = randval_mersenne(0, 7);
    switch (rand) {
    case 0:
        return VSUMW;
        break;
    case 1:
        return V_M;
        break;
    case 2:
        return VprW;
        break;
    case 3:
        return VdivW;
        break;
    case 4:
        return V_mean;
        break;
    case 5:
        return V_std;
        break;
    case 6:
        return VscalprW;
        break;
    case 7:
        return V_log;
        break;
    // case 7:
    //     return V_sum;
    //     break;
    // case 7:
    //     return V_sin;
    //     break;
    // case 10:
    //     return V_max;
    //     break;
    // case 11:
    //     return V_min;
    //     break;
    // case 8:
    //     return V_pow2;
    //     break;
    // case 9:
    //     return V_scalpow2;
    //     break;
    // case 9:
    //     return V_cos;
    //     break;
    default:
        return nullptr;
        break;
    }
}

struct Gene {
    Gene()
        : O(0x0)
    // , p(0)
    // , q(0)
    {
    }

    iptr O;

    int  instruction[ARITY + 1] = {0}; // list of registers (i.e. destination register & source register)
    bool is_effective           = false;
    // int p;                            // Starting (index or backwards steps) of the ranged aggregate
    // function int q;                   // Ending index or subgroup size of the ranged aggregate function
};

struct Individual {
    Individual()
        : fitness(MAXFLOAT)
        , best_register(-1)
        , Rp(1)
        , genes(randval_mersenne(1, GN))
    {
    }

    /// @brief Used for creating children in crossover @param size
    Individual(int size)
        : fitness(MAXFLOAT)
        , best_register(-1)
        , Rp(1)
        , genes(size)
    {
    }

    std::vector<Gene> genes;
    double            fitness;
    double            testing_fitness;
    int               best_register;

    std::vector<REGISTERS> Rp;
};
Individual  population[POP_SIZE];
Individual* best_individual = nullptr;

/// @brief Initialises the registers with input data and random constant registers
void init()
{
    // 1. Initialize the constants registers
    for (int j = registers; j < registers + CONSTS; j++)
        R[j].setConstant(randval_mersenne(CONST_MIN, CONST_MAX));

    // 2. Initialize the input registers from dataset. Resizes the registers if the input is not a vector
    for (int j = 0; j < registers; j++) {
        auto counter = 0;
        auto prev    = training_data(0, std::floor(j / REG_F), 0);
        for (auto i = 0; i < training_data.dimension(2); i++) {
            if (i + 1 != training_data.dimension(2)) {
                if (training_data(0, std::floor(j / REG_F), i + 1) == prev) {
                    counter++;
                }
            }
        }
        if (counter == training_data.dimension(2) - 1) {
            std::array<long, 3> offset  = {0, static_cast<long int>(std::floor(j / REG_F)), 0};
            std::array<long, 3> extents = {training_cases, 1, 1};
            std::array<long, 2> shape   = {training_cases, 1};
            R[j]                        = training_data.slice(offset, extents).reshape(shape);
            // std::cout << "R[j] = \n" << R[j] << std::endl;
        }
        else {
            std::array<long, 3> offset = {
                0, static_cast<long int>(std::floor(j / REG_F)), 0};   // Starting point
            std::array<long, 3> extent = {training_cases, 1, max_dim}; // Extent of the tensor
            std::array<long, 2> shape  = {training_cases, max_dim};
            R[j]                       = training_data.slice(offset, extent).reshape(shape);
            // std::cout << "R[j] = \n" << R[j] << std::endl;
        }
    }
    // 3. Initialize the constants registers for testing
    for (int j = registers; j < registers + CONSTS; j++)
        S[j].setConstant(randval_mersenne(CONST_MIN, CONST_MAX));

    // 4. Initialize the input resgisters for testing from dataset
    for (int j = 0; j < registers; j++) {
        auto counter = 0;
        auto prev    = testing_data(0, std::floor(j / REG_F), 0);
        for (auto i = 0; i < testing_data.dimension(2); i++) {
            if (i + 1 != testing_data.dimension(2)) {
                if (testing_data(0, std::floor(j / REG_F), i + 1) == prev) {
                    counter++;
                }
            }
        }
        if (counter == testing_data.dimension(2) - 1) {
            std::array<long, 3> offset  = {0, static_cast<long int>(std::floor(j / REG_F)), 0};
            std::array<long, 3> extents = {testing_cases, 1, 1};
            std::array<long, 2> shape   = {testing_cases, 1};
            S[j]                        = testing_data.slice(offset, extents).reshape(shape);
            // std::cout << "S[j] = \n" << S[j] << std::endl;
        }
        else {
            std::array<long, 3> offset = {
                0, static_cast<long int>(std::floor(j / REG_F)), 0};  // Starting point
            std::array<long, 3> extent = {testing_cases, 1, max_dim}; // Extent of the tensor
            std::array<long, 2> shape  = {testing_cases, max_dim};
            S[j]                       = testing_data.slice(offset, extent).reshape(shape);
            // std::cout << "S[j] = \n" << S[j] << std::endl;
        }
    }
}

void init_individual(Individual& ind)
{
    for (int i = 0; i < ind.genes.size(); i++) {
        // 1. Randomly initialize the instructions
        ind.genes[i].O = rand_instruction();

        // 2. Set the destination register
        ind.genes[i].instruction[0] = randval_mersenne(0, registers - 1);

        // 3. Randomly initialize the operands (i.e register orders)
        for (int j = 1; j <= ARITY; j++) {
            ind.genes[i].instruction[j] = randval_mersenne(0, registers + CONSTS - 1);
            auto dbg                    = 0;
        }

        // 4. Randomly initialize the ranged aggregate function parameters
        // ind.genes[i].p = randval_mersenne(0, max_dim - 1);
        // ind.genes[i].q = randval_mersenne(0, max_dim - 1);
    }
}

void init_population()
{
    init();
    for (int i = 0; i < POP_SIZE; i++)
        init_individual(population[i]);
}

/// @brief Evaluates the fitness of the individual @param ind
void eval_fitness(Individual& ind, TYPE evaluation_type)
{
    std::array<int, 1> rmse_dimension_1 = {1}; // Axis to operate on. (vector dimensions)
    std::array<int, 1> rmse_dimension_2 = {0}; // Axis to operate on. (training cases)
    if (evaluation_type == TYPE::TRAINING)
        output = training_output;
    else
        output = testing_output;
    // 2. Calculate the fitness for each register OR take specific register for output only
    for (int j = 0; j < registers; j++) {
        // 2.1. Calculate the fitness
        double rmse;
        auto   predicted_dimensions = ind.Rp[j].dimensions().back();
        auto   actual_dimensions    = output.dimensions().back();
        MATRIX diff(training_cases, 1);
        if (predicted_dimensions < actual_dimensions) {
            diff.resize(training_cases, actual_dimensions);
            for (auto i = 0; i < training_cases; i++) {
                for (auto k = 0; k < actual_dimensions; k++) {
                    diff(i, k) = ind.Rp[j](i, 0) - output(i, k);
                }
            }
        }
        else if (predicted_dimensions > actual_dimensions) {
            diff.resize(training_cases, predicted_dimensions);
            for (auto i = 0; i < training_cases; i++) {
                for (auto k = 0; k < predicted_dimensions; k++) {
                    diff(i, k) = ind.Rp[j](i, k) - output(i, 0);
                }
            }
        }
        else {
            diff.resize(training_cases, predicted_dimensions);
            diff = ind.Rp[j] - output;
        }
        Eigen::Tensor<double, 2> squared = diff.square();
        if (predicted_dimensions > 1) {
            Eigen::Tensor<double, 1> summed   = squared.sum(rmse_dimension_1);
            Eigen::Tensor<double, 1> vec_rmse = ((summed / (double)predicted_dimensions).sqrt());
            auto                     mean     = static_cast<Eigen::Tensor<double, 0>>(vec_rmse.mean())(0);
            vec_rmse                          = vec_rmse - mean;
            auto vec_rmse_squared             = vec_rmse.square();
            auto vec_rmse_summed              = vec_rmse_squared.sum(rmse_dimension_2);

            rmse =
                static_cast<Eigen::Tensor<double, 0>>((vec_rmse_summed / (double)training_cases).sqrt())(0);
        }
        else {
            auto summed = squared.sum(rmse_dimension_2);
            // std::cout << "Summed: \n" << summed << std::endl;
            rmse = std::sqrt(static_cast<Eigen::Tensor<double, 1>>(summed)(0) / (double)training_cases);
            // rmse = tensor_rmse(0);
        }
        if (predicted_dimensions != actual_dimensions)
            rmse += 500.0; // Punish for wrong dimensionality

        // Eigen::Tensor<double, 2> squared = diff.square();
        // // std::cout << "Squared: \n" << squared << std::endl;
        // Eigen::Tensor<double, 1> summed = squared.sum(rmse_dimension_1);
        // // std::cout << "Summed: \n" << summed << std::endl;
        // Eigen::Tensor<double, 1> rmse_vec = (summed / (double)max_dim).sqrt();
        // // std::cout << "RMSE vec: \n" << rmse_vec << std::endl;
        // auto mean_rmse = static_cast<Eigen::Tensor<double, 0>>(rmse_vec.mean())(0);
        // std::cout << "Mean RMSE: " << mean_rmse << std::endl;
        // Eigen::Tensor<double, 1> diff_rmse = rmse_vec - mean_rmse;
        // std::cout << "Diff RMSE: \n" << diff_rmse << std::endl;
        // Eigen::Tensor<double, 1> squared_rmse = diff_rmse.square();
        // std::cout << "Squared RMSE: \n" << squared_rmse << std::endl;
        // Eigen::Tensor<double, 0> summed_rmse = squared_rmse.sum(rmse_dimension_2);
        // std::cout << "Summed RMSE: \n" << summed_rmse << std::endl;
        // if (evaluation_type == TYPE::TRAINING) {
        //     // rmse = static_cast<Eigen::Tensor<double, 0>>((summed_rmse /
        //     (double)training_cases).sqrt())(0);
        //     // auto summed_rmse = static_cast<Eigen::Tensor<double, 0>>(rmse_vec.sum(rmse_dimension_2))(0);
        //     // rmse             = std::sqrt(summed_rmse / training_cases);
        //     rmse = mean_rmse;
        // }
        // else {
        //     // rmse = static_cast<Eigen::Tensor<double, 0>>((summed_rmse /
        //     (double)testing_cases).sqrt())(0);
        //     // auto summed_rmse = static_cast<Eigen::Tensor<double, 0>>(rmse_vec.sum(rmse_dimension_2))(0);
        //     // rmse             = std::sqrt(summed_rmse / training_cases);
        //     rmse = mean_rmse;
        // }

        if (evaluation_type == TYPE::TRAINING) {
            if (rmse < ind.fitness) {
                ind.fitness       = rmse;
                ind.best_register = static_cast<int>(std::floor(j / REG_F));
            }
            g_fitness_mutex.lock();
            if (rmse < best_fitness) {
                best_fitness = rmse;
                if (best_individual == nullptr) {
                    best_individual = new Individual(ind);
                }
                else {
                    delete best_individual;
                    best_individual = new Individual(ind);
                }
            }
            g_fitness_mutex.unlock();
        }
        // Update the testing fitness
        else {
            // No need to acquire lock here because this is only called by the main thread
            best_individual->testing_fitness = rmse;
        }
    }
}

/// @brief Computes the individual @param ind @todo Use CUDA for GPU acceleration
void compute(Individual& ind, TYPE type)
{
    // Reset the registers for calculations
    if (type == TYPE::TRAINING)
        ind.Rp = R;
    else
        ind.Rp = S;
    for (int j = 0; j < ind.genes.size(); j++) {
        // Execute the program. If using more then arity 2, need to loop
        auto dest    = ind.genes[j].instruction[0];
        auto op1     = ind.genes[j].instruction[1];
        auto op2     = ind.genes[j].instruction[2];
        ind.Rp[dest] = (ind.genes[j].O)(ind.Rp.at(op1), ind.Rp.at(op2));
    }
}

/// @brief  Executes the instructions of each individual in the population
void objective()
{
    for (int i = 0; i < POP_SIZE; i++) {
        // 1. Execute the intruction of the individual.
        compute(population[i], TYPE::TRAINING);
        // 2. Calculate the fitness for each register OR take specific register for output only
        eval_fitness(population[i], TYPE::TRAINING);
    }
}

/// @brief Custom micro mutation the individual of the given index @param parents
void custom_micro_mutation(std::vector<Individual>& parents)
{
    for (auto i = 0; i < 2; i++) {
        // 1. For each selected individual, randomly select a gene to mutate
        auto gene = randval_mersenne(0, parents[i].genes.size() - 1);
        // 2.1 With a proba of F mutate the operation
        if (randval_mersenne(0.0f, 1.0f) < F_MICRO) {
            parents[i].genes[gene].O = rand_instruction();
        }
        // 2.2. With a probability of F, mutate the destination register
        if (randval_mersenne(0.0f, 1.0f) < F_MICRO) {
            parents[i].genes[gene].instruction[0] = randval_mersenne(0, registers - 1);
        }
        // 2.3. With a probability of F, mutate the operands
        for (int j = 1; j <= ARITY; j++) {
            if (randval_mersenne(0.0f, 1.0f) < F_MICRO) {
                parents[i].genes[gene].instruction[j] = randval_mersenne(0, registers + CONSTS - 1);
            }
        }
        // 2.4. With a probabili of F, mutate the constant
        if (randval_mersenne(0.0f, 1.0f) < F_MICRO) {
            for (auto j = 0; j < ARITY + 1; j++) {
                if (parents.at(i).genes.at(gene).instruction[j] >= registers) {
                    parents.at(i).genes.at(gene).instruction[j] = randval_mersenne(CONST_MIN, CONST_MAX);
                }
            }
        }
    }
}

/// @brief Original micro mutation from the book @param parents @param gene
void micro_mutation(std::vector<Individual>& parents)
{
    for (auto i = 0; i < 2; i++) {
        if (randval_mersenne(0.0, 1.0) <= F_MICRO) {
            // 1. For each parents, randomly select an effective gene/instruction to mutate
            auto gene = randval_mersenne(0, parents[i].genes.size() - 1);
            // Check if there is an effective instruction
            std::vector<int> effectives;
            effectives.reserve(parents[i].genes.size());
            for (auto g = 0; g < parents[i].genes.size(); g++) {
                if (parents[i].genes[g].instruction[0] == parents[i].best_register)
                    effectives.push_back(g);
            }
            if (!effectives.empty())
                gene = effectives[randval_mersenne(0, effectives.size() - 1)];
            // 2. Randomly select a mutation type register|operator|constant
            int  mutation_type;
            auto has_constant = false;
            for (auto j = 0; j < ARITY; j++) {
                if (parents.at(i).genes.at(gene).instruction[j] >= registers) { // If constant
                    has_constant = true;
                    break;
                }
            }
            has_constant ? mutation_type = randval_mersenne(0, 2) : mutation_type = randval_mersenne(0, 1);
            // 3.1. If mutation type is register, choose whether dest or op registers are to be mutated
            if (mutation_type == 0) {
                // Dest == 0, Op1 == 1, Op2 == 2
                int register_to_mutate;
                do {
                    register_to_mutate = randval_mersenne(0, ARITY);
                } while (register_to_mutate == parents[i].genes[gene].instruction[0] && !effectives.empty());
                // Mutate the destination register
                if (register_to_mutate == 0)
                    parents[i].genes[gene].instruction[0] = randval_mersenne(0, registers - 1);

                // Mutate the operands registers
                else {
                    parents[i].genes[gene].instruction[register_to_mutate] =
                        randval_mersenne(0, registers + CONSTS - 1);
                }
            }
            // 3.2. If mutation type is operator, mutate the operation
            if (mutation_type == 1)
                parents[i].genes[gene].O = rand_instruction();
            // 3.3. If mutation type is constant, mutate the operands
            if (mutation_type == 2) {
                for (auto j = 0; j < ARITY; j++) {
                    if (parents.at(i).genes.at(gene).instruction[j] >= registers) { // If constant
                        parents.at(i).Rp.at(j).setConstant(randval_mersenne(CONST_MIN, CONST_MAX));
                        // parents.at(i).flag = 0;
                    }
                }
            }
        }
    }
}

/// @brief Change the overall length by +-1 of the parents, right before reproduction @param parents
/// @return The only gene with the effective instruction
void macro_mutation(std::vector<Individual>& parents)
{
    for (auto i = 0; i < 2; i++) {
        if (randval_mersenne(0.0, 1.0) < F_MACRO) {
            // 1. Random mutation type determined by F_MACRO_INS
            // and F_MACRO_DEL
            auto proba = randval_mersenne(0.0, 1.0);
            // 2. Randomly select a mutation point
            auto p = randval_mersenne(0, parents[i].genes.size() - 1);
            // 3.1. If insertion and the length of the program is less than GN-1, insert a new gene
            if (proba < F_MACRO_INS && parents[i].genes.size() < GN - 1) {
                auto gene = Gene();
                // 3.1.1. Randomly set the destination register or set it to the best register
                gene.instruction[0] = randval_mersenne(0, registers - 1);
                // 3.1.2. Randomly set the operation
                gene.O = rand_instruction();
                // 3.1.3 Randomly initialize the operands (i.e register orders)
                for (int j = 1; j <= ARITY; j++)
                    gene.instruction[j] = randval_mersenne(0, registers + CONSTS - 1);
                // 3.1.4. Insert the gene
                parents[i].genes.insert(parents[i].genes.begin() + p, std::move(gene));
            }
            else if (proba > F_MACRO_INS && parents[i].genes.size() > 1) {
                // 3.2. If deletion and the length of the program is greater than 1, delete a gene
                parents[i].genes.erase(parents[i].genes.begin() + p);
            }
        }
        else
            continue;
    }
}

/// @brief Two point crossover @param winners 2 winners (parents) @param global_children Global list of
/// clidren
void crossover(const std::vector<int>& winners, std::vector<Individual>& global_children)
{
    // 1. Perform crossover with a probability of 0.9
    auto                    rand = randval_mersenne(0.0f, 1.0f);
    std::vector<Individual> parents(2);
    parents[0] = population[winners[0]];
    parents[1] = population[winners[1]];
    if (rand < CR) {
        // Crossover occurs

        // 1. Macro mutation
        macro_mutation(parents);

        // 2. In parallel micro mutate the selected individuals with some probability
        std::thread concurrent_mutation(micro_mutation, std::ref(parents));
        auto        crossover_point1_1 =
            randval_mersenne(0, parents[0].genes.size() - 1); // Parent 1 crossover points
        auto crossover_point1_2 = randval_mersenne(0, parents[0].genes.size() - 1);

        auto crossover_point2_1 =
            randval_mersenne(0, parents[1].genes.size() - 1); // Parent 2 crossover points
        auto crossover_point2_2 = randval_mersenne(0, parents[1].genes.size() - 1);

        // 3.1. Make sure crossover points are in order
        if (crossover_point1_1 > crossover_point1_2) {
            std::swap(crossover_point1_1, crossover_point1_2);
        }
        if (crossover_point2_1 > crossover_point2_2) {
            std::swap(crossover_point2_1, crossover_point2_2);
        }

        // 3.2. Set size of children
        int child1_size = (crossover_point1_1 + ((parents[0].genes.size() - 1) - crossover_point1_2)) +
                          (crossover_point2_2 - crossover_point2_1 + 1);
        int child2_size = (crossover_point2_1 + ((parents[1].genes.size() - 1) - crossover_point2_2)) +
                          (crossover_point1_2 - crossover_point1_1 + 1);

        // 3.3. Check that it doesn't exceed the maximum size
        if (child1_size > GN) {
            crossover_point2_2 = crossover_point2_1 + (crossover_point1_2 - crossover_point1_1);
            child1_size        = (crossover_point1_1 + ((parents[0].genes.size() - 1) - crossover_point1_2)) +
                          (crossover_point2_2 - crossover_point2_1 + 1);
            child2_size = (crossover_point2_1 + ((parents[1].genes.size() - 1) - crossover_point2_2)) +
                          (crossover_point1_2 - crossover_point1_1 + 1);
        }
        if (child2_size > GN) {
            crossover_point1_2 = crossover_point1_1 + (crossover_point2_2 - crossover_point2_1);
            child2_size        = (crossover_point2_1 + ((parents[1].genes.size() - 1) - crossover_point2_2)) +
                          (crossover_point1_2 - crossover_point1_1 + 1);
            child1_size = (crossover_point1_1 + ((parents[0].genes.size() - 1) - crossover_point1_2)) +
                          (crossover_point2_2 - crossover_point2_1 + 1);
        }

        auto child1 = Individual(child1_size);
        auto child2 = Individual(child2_size);

        concurrent_mutation.join();

        // 3.4.1 Copy first portion from first parent
        std::copy(
            parents[0].genes.begin(), parents[0].genes.begin() + crossover_point1_1, child1.genes.begin());
        // 3.4.2 Copy second portion from the second parent
        std::copy(parents[1].genes.begin() + crossover_point2_1,
            parents[1].genes.begin() + crossover_point2_2 + 1, child1.genes.begin() + crossover_point1_1);
        // 3.4.3 Copy third portion from first parent
        std::copy(parents[0].genes.begin() + crossover_point1_2 + 1, parents[0].genes.end(),
            child1.genes.begin() + crossover_point1_1 + (crossover_point2_2 - crossover_point2_1 + 1));

        // 3.5.1 Copy first portion from second parent
        std::copy(
            parents[1].genes.begin(), parents[1].genes.begin() + crossover_point2_1, child2.genes.begin());
        // 3.5.2 Copy second portion from the first parent
        std::copy(parents[0].genes.begin() + crossover_point1_1,
            parents[0].genes.begin() + crossover_point1_2 + 1, child2.genes.begin() + crossover_point2_1);
        // 3.5.3 Copy third portion from second parent
        std::copy(parents[1].genes.begin() + crossover_point2_2 + 1, parents[1].genes.end(),
            child2.genes.begin() + crossover_point2_1 + (crossover_point1_2 - crossover_point1_1 + 1));

        // // 3.4. Copy the genes from the parents to the children
        // for (auto gene = 0; gene < child1_size; gene++) {
        //     // 3.4.1 Copy first portion from first parent
        //     if (gene < crossover_point1_1) {
        //         child1.genes[gene] = parents[0].genes[gene];
        //     }
        //     // 3.4.2 Copy second portion from the second parent
        //     else if (gene == crossover_point1_1) {
        //         for (auto g = crossover_point2_1; g <= crossover_point2_2; g++) {
        //             child1.genes.at(gene) = parents[1].genes.at(g);
        //             gene++;
        //         }
        //         gene--; // Go back one iteration (because gene gets incremented twice)
        //     }
        //     // 3.4.3 Copy last portion from the first parent again
        //     else {
        //         for (auto g = crossover_point1_2; g < parents[0].genes.size() - 1; g++) {
        //             child1.genes.at(gene) = parents[0].genes.at(g);
        //             gene++;
        //         }
        //         break;
        //     }
        // }
        // for (auto gene = 0; gene < child2_size; gene++) {
        //     if (gene < crossover_point2_1) {
        //         child2.genes[gene] = parents[1].genes[gene];
        //     }
        //     else if (gene == crossover_point2_1) {
        //         for (auto g = crossover_point1_1; g <= crossover_point1_2; g++) {
        //             child2.genes.at(gene) = parents[0].genes.at(g);
        //             gene++;
        //         }
        //         gene--;
        //     }
        //     else {
        //         for (auto g = crossover_point2_2; g < parents[1].genes.size() - 1; g++) {
        //             child2.genes.at(gene) = parents[1].genes.at(g);
        //             gene++;
        //         }
        //         break;
        //     }
        // }

        // 3.5. Execute the instructions of the children
        compute(child1, TYPE::TRAINING);
        eval_fitness(child1, TYPE::TRAINING);
        compute(child2, TYPE::TRAINING);
        eval_fitness(child2, TYPE::TRAINING);

        // 3.6. Add the new children to the global children container after acquiring mutex
        g_children_mutex.lock();
        global_children.emplace_back(std::move(child1));
        global_children.emplace_back(std::move(child2));
        g_children_mutex.unlock();
    }
    else {
        // No crossover occurs
        // Mutations may still occur
        macro_mutation(parents);
        micro_mutation(parents);  // Mutate the 2 parents
        auto child1 = parents[0]; // child1 = parent1
        auto child2 = parents[1]; // child2 = parent2

        compute(child1, TYPE::TRAINING);
        eval_fitness(child1, TYPE::TRAINING);
        compute(child2, TYPE::TRAINING);
        eval_fitness(child2, TYPE::TRAINING);

        g_children_mutex.lock();
        global_children.emplace_back(std::move(child1));
        global_children.emplace_back(std::move(child2));
        g_children_mutex.unlock();
    }
}

/// @brief Performs 2 tournaments to get 2 winners as the parents
void selection(std::unordered_map<int, int>& global_losers, std::vector<Individual>& children)
{
    std::vector<int> curr_losers;
    std::vector<int> curr_winners;
    curr_losers.reserve((TOURNAMENT_SIZE - 1) * 2);
    curr_winners.reserve(2);

    // Do 2 tournaments to get 2 parents/winners
    for (auto j = 0; j < 2; j++) {
        auto                         best = MAXFLOAT; // Best fitness in the current tournament
        int                          best_index;
        std::unordered_map<int, int> tournament;

        for (auto k = 0; k < TOURNAMENT_SIZE; k++) {
            // Select random population for tournament
            auto rand_ind = randval_mersenne(0, POP_SIZE - 1);
            // Make sure we pick different individuals
            if (tournament.contains(rand_ind)) {
                auto already_picked_ind = rand_ind;
                do {
                    rand_ind = randval_mersenne(0, POP_SIZE - 1);
                } while (rand_ind == already_picked_ind);
                tournament[rand_ind] = 1;
            }
            else {
                tournament[rand_ind] = 1;
            }
            if (population[rand_ind].fitness < best) {
                // Add prev best to losers
                if (best != MAXFLOAT) {
                    curr_losers.push_back(rand_ind);
                }
                best       = population[rand_ind].fitness;
                best_index = rand_ind;
            }
            else {
                curr_losers.push_back(rand_ind);
            }
        }
        curr_winners.push_back(best_index);
    }
    // Update global losers after acquiring the mutex
    g_losers_mutex.lock();
    for (auto& l : curr_losers)
        global_losers[l] = 1;
    g_losers_mutex.unlock();
    crossover(curr_winners, children); // Curr winners are the parents
}

/// @brief 1. Selection 2. Crossover 3. Mutation (in the crossover function)
void production()
{
    auto it = static_cast<int>(POP_SIZE / 2); // Number of iterations (/2 because crossover makes 2 children)

    std::vector<Individual> children; // Global children for this generation
    children.reserve(POP_SIZE);
    std::unordered_map<int, int> losers; // non-repeating global losers
    losers.reserve((POP_SIZE * (TOURNAMENT_SIZE - 1)) / TOURNAMENT_SIZE);

    for (auto i = 0; i < it; i++) {
        // 1. Do 2 tournaments to get 2 parents/winners. Add selection/cross/mutation tasks to the pool
        for (auto j = 0; j < concurrent_tournaments.get_thread_count() && i < it; j++) {
            concurrent_tournaments.push_task(selection, std::ref(losers), std::ref(children));
            i++;
        }
    }
    concurrent_tournaments.wait_for_tasks(); // Wait for the currently running tasks to finish
    // 2. Replace the losers from the old generation with the new one
    for (auto& loser : losers) {
        auto random_child       = std::next(std::begin(children), randval_mersenne(0, children.size() - 1));
        population[loser.first] = *random_child;
        children.erase(random_child); // Make sure we don't include the same child multiple times
        if (children.empty())
            break;
    }
}

void validate()
{
    best_individual->Rp = S;
    // 1. Compute the best individual on unseen data
    compute(*best_individual, TYPE::TESTING);
    // 2. Calculate the fitness for each register
    eval_fitness(*best_individual, TYPE::TESTING);
    // 3. Save test results
    testing_fitness.emplace_back(best_individual->testing_fitness);
}

void evolve()
{
    // 1.  Initial execution of all programs and fitness evaluation
    objective();
    best_individuals.reserve(POP_SIZE);
    testing_fitness.reserve(POP_SIZE);

    // Loop untill reached number of generations or perfect hit
    for (auto g = 0; g < MAX_GEN; g++) {
        // 2. Tournament selection + crossover + mutation
        production();
        // 3. Add the best individual of current generation to the vector of best individuals
        best_individuals.emplace_back(best_individual->fitness);
        // 4. If perfect hit exit
        if (best_individual != nullptr)
            if (best_individual->fitness <= 1e-4) {
                validate(); // Before exiting, test the best individual on unseen data
                break;
            }
        // 5. Validate the best individual of the current generation
        validate();
    }
}

/// @brief Prints the results (recursively) @param reg_index The register index we are evaluating  @param gene
void print_best(int reg_index, int gene)
{
    if (gene > 0) {
        // If the current gene has an effective expression
        if (best_individual->genes[gene].instruction[0] == reg_index) {
            if (best_individual->genes[gene].O == VSUMW)
                std::cout << "VSUMW(";
            else if (best_individual->genes[gene].O == V_M) {
                std::cout << "V_M(";
            }
            else if (best_individual->genes[gene].O == VprW) {
                std::cout << "VprW(";
            }
            else if (best_individual->genes[gene].O == VdivW) {
                std::cout << "VdivW(";
            }
            else if (best_individual->genes[gene].O == V_mean) {
                std::cout << "V_mean(";
            }
            else if (best_individual->genes[gene].O == V_std) {
                std::cout << "V_std(";
            }
            else if (best_individual->genes[gene].O == VscalprW) {
                std::cout << "VscalprW(";
            }
            else if (best_individual->genes[gene].O == V_sum) {
                std::cout << "V_sum(";
            }
            else if (best_individual->genes[gene].O == V_min) {
                std::cout << "V_min(";
            }
            else if (best_individual->genes[gene].O == V_max) {
                std::cout << "V_max(";
            }
            else if (best_individual->genes[gene].O == V_sin) {
                std::cout << "V_sin(";
            }
            else if (best_individual->genes[gene].O == V_log) {
                std::cout << "V_log(";
            }
            else if (best_individual->genes[gene].O == V_scalpow2) {
                std::cout << "V_scalpow2(";
            }
            else if (best_individual->genes[gene].O == V_pow2) {
                std::cout << "V_pow2(";
            }
            print_best(best_individual->genes[gene].instruction[1], gene - 1);
            std::cout << ", ";
            print_best(best_individual->genes[gene].instruction[2], gene - 1);
            std::cout << ")";
        }
        else
            print_best(reg_index, gene - 1);
    }
    else if (gene == 0) {
        if (reg_index < registers)
            std::cout << "X" << static_cast<int>(std::floor(reg_index / REG_F));
        else
            std::cout << "Rc(" << best_individual->Rp.at(reg_index)(0, 0) << ")";
        // Could output real value instead of arbitrary C
    }
    else
        return;
}

int main(int argc, char const* argv[])
{
    // --------------------------------------------------------------------------------------------
    // Read the data and update the corresponding global vars
    read_data(BENCHMARK, training_data, training_output, training_cases, max_dim, max_dim_out, registers,
        testing_data, testing_output, testing_cases);
    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    // Scale the registers
    registers = registers * REG_F;

    R.resize(registers + CONSTS);
    S.resize(registers + CONSTS);
    // Resize registers to fit vectors (the actual size is modified dynamically at runtime)
    for (auto r = 0; r < registers; r++) {
        R.at(r).resize(training_cases, max_dim);
        S.at(r).resize(training_cases, max_dim);
    }
    // Set the constant registers size (the size is fixed)
    for (auto c = 0; c < CONSTS; c++) {
        R.at(registers + c).resize(training_cases, 1);
        S.at(registers + c).resize(training_cases, 1);
    }

    for (auto i = 0; i < RUNS; i++) {
        // Start the timer
        // auto start = std::chrono::high_resolution_clock::now();

        // 1. Initialize population
        init_population();

        // 2. Evolve population
        evolve();

        // auto stop     = std::chrono::high_resolution_clock::now(); // Stop the timer
        // auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

        // 3. Display/save results

        // Evaluate the evolved program and print it
        std::cout << std::endl;
        print_best(best_individual->best_register, best_individual->genes.size() - 1);
        std::cout << "\nBest individual program length: " << best_individual->genes.size() << std::endl;
        // Write best fitness
        std::cout << "Best fitness: " << best_fitness << std::endl;
        // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
        // Write best individuals to csv file
        std::ofstream best_individuals_file;
        best_individuals_file.open("../results/F_" + std::to_string(BENCHMARK) + "_G" +
                                       std::to_string(MAX_GEN) + "_P" + std::to_string(POP_SIZE) +
                                       "-best_individuals.csv",
            std::ios::out | std::ios::app);
        for (auto bi : best_individuals)
            best_individuals_file << bi << ",";
        best_individuals_file << '\n';
        best_individuals_file.close();
        // Write testing fitness to csv file
        std::ofstream testing_fitness_file;
        testing_fitness_file.open("../results/F_" + std::to_string(BENCHMARK) + "_G" +
                                      std::to_string(MAX_GEN) + "_P" + std::to_string(POP_SIZE) +
                                      "-testing_fitness.csv",
            std::ios::out | std::ios::app);
        for (auto tf : testing_fitness)
            testing_fitness_file << tf << ",";
        testing_fitness_file << '\n';
        testing_fitness_file.close();

        // Reset for statistics
        for (auto j = 0; j < POP_SIZE; j++) {
            population[j] = Individual();
        }
        best_fitness = MAXFLOAT;
        best_individuals.clear();
        testing_fitness.clear();
    }

    // 4. Clean up
    if (best_individual != nullptr)
        delete best_individual;

    return 0;
}