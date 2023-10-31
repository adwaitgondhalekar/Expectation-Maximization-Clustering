#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <random>
#include <chrono>
#include <Eigen/Dense>
using namespace std;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

int dimensions = 4;
int gaussians = 3;
int total_rows = 150;
int total_columns = 5;
int iterations = 100;
double tolerance = pow(10, -4);
MatrixXd observation(total_rows, (dimensions + 1));
MatrixXd observation_unlabelled(150, (dimensions));
MatrixXd observation_unlabelled_zero_mean(150, (dimensions));
MatrixXd observation_1(50, (dimensions));
MatrixXd observation_2(50, (dimensions));
MatrixXd observation_3(50, (dimensions));
VectorXd mean_gaussian1(dimensions);
VectorXd mean_gaussian2(dimensions);
VectorXd mean_gaussian3(dimensions);
VectorXd mean_full_data(dimensions);
VectorXd std_dev_gaussian1;
VectorXd std_dev_gaussian2;
VectorXd std_dev_gaussian3;
Vector3d prior_probability;
MatrixXd gaussian1_covariance((dimensions), (dimensions));
MatrixXd gaussian2_covariance((dimensions), (dimensions));
MatrixXd gaussian3_covariance((dimensions), (dimensions));
VectorXd prob_x_given_gaussian1(total_rows);
VectorXd prob_x_given_gaussian2(total_rows);
VectorXd prob_x_given_gaussian3(total_rows);
VectorXd posterior_prob_gaussian1(total_rows);
VectorXd posterior_prob_gaussian2(total_rows);
VectorXd posterior_prob_gaussian3(total_rows);
VectorXd data_point_weight_gaussian1(total_rows);
VectorXd data_point_weight_gaussian2(total_rows);
VectorXd data_point_weight_gaussian3(total_rows);
double sum_posterior_prob_gaussian1 = 0;
double sum_posterior_prob_gaussian2 = 0;
double sum_posterior_prob_gaussian3 = 0;
VectorXd likelihood_mul_prior_gaussian1(total_rows);
VectorXd likelihood_mul_prior_gaussian2(total_rows);
VectorXd likelihood_mul_prior_gaussian3(total_rows);
double prev_log_likelihood = 0;
double next_log_likelihood = 0;
VectorXd actual_cluster(total_rows);
VectorXd allotted_cluster(total_rows);

void read_csv_data(string filename)
{

    fstream file;
    string line;
    file.open(filename, ios::in | ios::out);
    int obs_no = 0;
    while (file)
    {
        getline(file, line);
        istringstream is(line);
        int params = 0;
        int col = 0;
        string data = "";
        vector<float> data_point;
        while (getline(is, data, ','))
        {
            if (params < (dimensions))
            {
                observation(obs_no, params) = stod(data);
                observation_unlabelled(obs_no, params) = stod(data);
                col++;
            }
            else if (params == (total_columns - 1))
            {
                if (data.compare("Iris-setosa") == 0)
                {
                    observation(obs_no, col) = 0;
                    actual_cluster(obs_no) = 0;
                }

                else if (data.compare("Iris-versicolor") == 0)
                {
                    observation(obs_no, col) = 1;
                    actual_cluster(obs_no) = 1;
                }

                else
                {
                    observation(obs_no, col) = 2;
                    actual_cluster(obs_no) = 2;
                }
            }

            params++;
        }
        // observation.push_back(data_point);
        obs_no++;
    }
}
void assign_points_to_gaussians()
{
    for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < (dimensions); j++)
        {
            observation_1(i, j) = observation_unlabelled(i, j);
        }
    }

    for (int i = 50; i < 100; i++)
    {
        for (int j = 0; j < (dimensions); j++)
        {
            observation_2(i - 50, j) = observation_unlabelled(i, j);
        }
    }

    for (int i = 100; i < 150; i++)
    {
        for (int j = 0; j < (dimensions); j++)
        {
            observation_3(i - 100, j) = observation_unlabelled(i, j);
        }
    }
}

// void get_initial_gaussian_means(int gaussians)
// {
//     // choosing 3 random data points from the dataset and assigning them as the initial means for the gaussians
//     srand(time(0));
//     for (int i = 0; i < gaussians; i++)
//     {
//         int j = i + rand() % (150 - i);
//         // cout<<j<<endl;
//         if (i == 0)
//             mean_gaussian1 = observation_unlabelled.row(j);
//         if (i == 1)
//             mean_gaussian2 = observation_unlabelled.row(j);
//         if (i == 2)
//             mean_gaussian3 = observation_unlabelled.row(j);

//     }
// }

// randomized initial covariance function
// void randomized_initial_covariance()
// {

//     srand(time(0));
//     gaussian1_covariance.setRandom();
//     srand(time(0));
//     gaussian2_covariance.setRandom();
//     srand(time(0));
//     gaussian3_covariance.setRandom();
// }

void get_gaussian_initial_mean()
{

    // getting mean of attributes for 0-49 observations (examples belonging to cluster 0)
    mean_gaussian1 = observation_1.colwise().mean();

    // getting mean of attributes for 50-99 observations (examples belonging to cluster 1)
    mean_gaussian2 = observation_2.colwise().mean();

    // getting mean of attributes for 100-149 observations (examples belonging to cluster 2)
    mean_gaussian3 = observation_3.colwise().mean();

    // calculating mean of the complete dataset (over examples of all the clusters)
    mean_full_data = observation_unlabelled.colwise().mean();
}

void initialize_gaussian_priors()
{
    // initializing priors for each gaussian

    Vector3d prior;
    for (int i = 0; i < gaussians; i++)
    {
        // asigning equal priors to all gaussians
        prior[i] = 1;
    }
    prior = prior / gaussians;
    prior_probability = prior;
}

// // old initial covariance function

// void get_initial_covariance(MatrixXd m1, int gaussian)
// {

//     for (int i = 0; i < (dimensions); i++)
//     {

//         for (int j = 0; j < (dimensions); j++)
//         {
//             switch (gaussian)
//             {
//             case 1:
//                 gaussian1_covariance(i, j) = m1.col(i).dot(m1.col(j));
//                 break;
//             case 2:
//                 gaussian2_covariance(i, j) = m1.col(i).dot(m1.col(j));
//                 break;
//             case 3:
//                 gaussian3_covariance(i, j) = m1.col(i).dot(m1.col(j));
//                 break;
//             default:
//                 break;
//             }
//         }
//     }

//     if (gaussian == 1)
//         gaussian1_covariance /= (50);
//     else if (gaussian == 2)
//         gaussian2_covariance /= (50);
//     else
//         gaussian3_covariance /= (50);
// }

// new initial covariance function

void get_initial_covariance()
{

    // subtracting mean (of complete dataset) from each observation in the dataset

    for (int i = 0; i < observation_unlabelled.rows(); i++)
    {
        observation_unlabelled_zero_mean.row(i) = observation_unlabelled.row(i) - (mean_full_data.transpose());
    }

    for (int i = 0; i < (dimensions); i++)
    {

        for (int j = 0; j < (dimensions); j++)
        {

            gaussian1_covariance(i, j) = observation_unlabelled_zero_mean.col(i).dot(observation_unlabelled_zero_mean.col(j));

            gaussian2_covariance(i, j) = observation_unlabelled_zero_mean.col(i).dot(observation_unlabelled_zero_mean.col(j));

            gaussian3_covariance(i, j) = observation_unlabelled_zero_mean.col(i).dot(observation_unlabelled_zero_mean.col(j));
        }
    }

    gaussian1_covariance /= (observation_unlabelled_zero_mean.rows() - 1);
    gaussian2_covariance /= (observation_unlabelled_zero_mean.rows() - 1);
    gaussian3_covariance /= (observation_unlabelled_zero_mean.rows() - 1);
}

void multivariate_gaussian_distribution(MatrixXd obs, MatrixXd mean, MatrixXd cov, int gaussian)
{
    int obs_no = 0;
    for (auto row : obs.rowwise())
    {

        MatrixXd mean_subtracted = row - mean.transpose();
        MatrixXd product = mean_subtracted * (cov.inverse());
        MatrixXd exp_input = product * (mean_subtracted.transpose());

        switch (gaussian)
        {
        case 0:
        {
            // using formula of multivariate gaussian distribution
            prob_x_given_gaussian1[obs_no] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((cov.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
            break;
        }

        case 1:
        {
            // using formula of multivariate gaussian distribution
            prob_x_given_gaussian2[obs_no] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((cov.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
            break;
        }

        case 2:
        {
            // using formula of multivariate gaussian distribution
            prob_x_given_gaussian3[obs_no] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((cov.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
            break;
        }

        default:
            break;
        }
        obs_no++;
    }
}

void calculate_posterior_probability(MatrixXd obs, VectorXd prior, VectorXd likelihood1, VectorXd likelihood2, VectorXd likelihood3, int gaussian)
{
    double numerator = 0;
    double denominator = 0;

    for (int i = 0; i < observation_unlabelled.rows(); i++)
    {

        denominator = (likelihood1[i] * prior[0]) + (likelihood2[i] * prior[1]) + (likelihood3[i] * prior[2]);

        switch (gaussian)
        {
        case 0:
        {
            numerator = likelihood1[i] * prior[gaussian];
            likelihood_mul_prior_gaussian1(i) = numerator;
            posterior_prob_gaussian1[i] = (numerator / denominator);
            sum_posterior_prob_gaussian1 += posterior_prob_gaussian1[i];
            break;
        }

        case 1:
        {
            numerator = likelihood2[i] * prior[gaussian];
            likelihood_mul_prior_gaussian2(i) = numerator;
            posterior_prob_gaussian2[i] = (numerator / denominator);
            sum_posterior_prob_gaussian2 += posterior_prob_gaussian2[i];
            break;
        }

        case 2:
        {
            numerator = likelihood3[i] * prior[gaussian];
            likelihood_mul_prior_gaussian3(i) = numerator;
            posterior_prob_gaussian3[i] = (numerator / denominator);
            sum_posterior_prob_gaussian3 += posterior_prob_gaussian3[i];
            break;
        }

        default:
            break;
        }
    }
}

void e_step()
{
    // finding how typical is Xi under a given gaussian (calculating for each gaussian)

    multivariate_gaussian_distribution(observation_unlabelled, mean_gaussian1, gaussian1_covariance, 0);
    multivariate_gaussian_distribution(observation_unlabelled, mean_gaussian2, gaussian2_covariance, 1);
    multivariate_gaussian_distribution(observation_unlabelled, mean_gaussian3, gaussian3_covariance, 2);

    // calculating posterior probabilities that, given a point what is the probability of it belonging to any one of the gaussians
    sum_posterior_prob_gaussian1 = 0;
    calculate_posterior_probability(observation_unlabelled, prior_probability, prob_x_given_gaussian1, prob_x_given_gaussian2, prob_x_given_gaussian3, 0);
    sum_posterior_prob_gaussian2 = 0;
    calculate_posterior_probability(observation_unlabelled, prior_probability, prob_x_given_gaussian1, prob_x_given_gaussian2, prob_x_given_gaussian3, 1);
    sum_posterior_prob_gaussian3 = 0;
    calculate_posterior_probability(observation_unlabelled, prior_probability, prob_x_given_gaussian1, prob_x_given_gaussian2, prob_x_given_gaussian3, 2);
}
void re_calculate_mean(MatrixXd data, VectorXd posterior_prob, double total_posterior_prob, int gaussian)
{
    MatrixXd new_matrix(total_rows, (dimensions));
    for (int i = 0; i < data.rows(); i++)
    {
        new_matrix.row(i) = posterior_prob[i] * (data.row(i));
    }

    switch (gaussian)
    {
    case 0:
    {

        VectorXd new_mean = (new_matrix.colwise().sum()) / (total_posterior_prob);
        mean_gaussian1 = new_mean;
        break;
    }
    case 1:
    {
        VectorXd new_mean = (new_matrix.colwise().sum()) / (total_posterior_prob);
        mean_gaussian2 = new_mean;
        break;
    }
    case 3:
    {
        VectorXd new_mean = (new_matrix.colwise().sum()) / (total_posterior_prob);
        mean_gaussian3 = new_mean;
        break;
    }

    default:
        break;
    }
}

void re_calculate_covarinces(MatrixXd obs, VectorXd posterior_prob, double total_posterior_prob, int gaussian)
{
    MatrixXd new_covariance(observation_unlabelled.cols(), observation_unlabelled.cols());
    for (int i = 0; i < obs.cols(); i++)
    {
        for (int j = 0; j < obs.cols(); j++)
        {
            double sum = 0;
            for (int k = 0; k < obs.rows(); k++)
            {
                sum += posterior_prob[k] * obs(k, i) * obs(k, j);
            }
            new_covariance(i, j) = sum;
        }
    }

    new_covariance = new_covariance / total_posterior_prob;

    switch (gaussian)
    {
    case 0:
    {
        gaussian1_covariance = new_covariance;
        break;
    }
    case 1:
    {
        gaussian2_covariance = new_covariance;
        break;
    }
    case 2:
    {
        gaussian3_covariance = new_covariance;
        break;
    }

    default:
        break;
    }
}

void re_calculate_prior(double sum_posterior_prob, int gaussian)
{
    switch (gaussian)
    {
    case 0:
    {
        prior_probability[0] = sum_posterior_prob / (observation.rows());
        break;
    }
    case 1:
    {
        prior_probability[1] = sum_posterior_prob / (observation.rows());
        break;
    }
    case 2:
    {
        prior_probability[2] = sum_posterior_prob / (observation.rows());
        break;
    }

    default:
        break;
    }
}
void m_step()
{
    // re - calculating the mean, prior probability and standard deviation for each gaussian

    re_calculate_mean(observation_unlabelled, posterior_prob_gaussian1, sum_posterior_prob_gaussian1, 0);
    re_calculate_mean(observation_unlabelled, posterior_prob_gaussian2, sum_posterior_prob_gaussian2, 1);
    re_calculate_mean(observation_unlabelled, posterior_prob_gaussian3, sum_posterior_prob_gaussian3, 2);

    // re - calculating the covariance matrix for each gaussian

    // zero mean observations for mean of gaussian 1
    MatrixXd zero_mean_observations1 = observation_unlabelled.rowwise() - mean_gaussian1.transpose();
    re_calculate_covarinces(zero_mean_observations1, posterior_prob_gaussian1, sum_posterior_prob_gaussian1, 0);

    // zero mean observations for mean of gaussian 2
    MatrixXd zero_mean_observations2 = observation_unlabelled.rowwise() - mean_gaussian2.transpose();
    re_calculate_covarinces(zero_mean_observations2, posterior_prob_gaussian2, sum_posterior_prob_gaussian2, 1);

    // zero mean observations for mean of gaussian 3
    MatrixXd zero_mean_observations3 = observation_unlabelled.rowwise() - mean_gaussian3.transpose();
    re_calculate_covarinces(zero_mean_observations3, posterior_prob_gaussian3, sum_posterior_prob_gaussian3, 2);

    // re - calculating the priors for each gaussian
    re_calculate_prior(sum_posterior_prob_gaussian1, 0);
    re_calculate_prior(sum_posterior_prob_gaussian2, 1);
    re_calculate_prior(sum_posterior_prob_gaussian3, 2);
}
double get_log_likelihood()
{
    // calculating log likelihood
    double sum = 0;
    for (int i = 0; i < observation.rows(); i++)
    {
        sum += log10(likelihood_mul_prior_gaussian1(i) + likelihood_mul_prior_gaussian2(i) + likelihood_mul_prior_gaussian3(i));
    }
    return sum;
}

bool check_stopping_condition()
{
    double prev_log_likelihood_avg = prev_log_likelihood / (observation.rows());
    double next_log_likelihood_avg = next_log_likelihood / (observation.rows());

    if (abs(prev_log_likelihood_avg - next_log_likelihood_avg) < tolerance)
        return true;
    else
        return false;
}
void shuffle_data()
{
    srand(time(0));
    double temp = 0;
    for (int i = 0; i < (total_rows - 1); i++)
    {
        int j = i + rand() % (150 - i);
        observation.row(i).swap(observation.row(j));
        observation_unlabelled.row(i).swap(observation_unlabelled.row(j));
        temp = actual_cluster[j];
        actual_cluster[j] = actual_cluster[i];
        actual_cluster[i] = temp;
    }
}

// not to use this function
// void predict_class(VectorXd data_point)
// {
//     MatrixXd mean_subtracted(gaussians, (dimensions));
//     MatrixXd gaussian_means(gaussians, (dimensions));
//     // cout<<"gaussian mean"<<endl;

//     gaussian_means.row(0) = mean_gaussian1.transpose();
//     gaussian_means.row(1) = mean_gaussian2.transpose();
//     gaussian_means.row(2) = mean_gaussian3.transpose();

//     for (int i = 0; i < mean_subtracted.rows(); i++)
//     {
//         if (i == 0)
//             mean_subtracted.row(i) = (data_point - mean_gaussian1).transpose();
//         if (i == 1)
//             mean_subtracted.row(i) = (data_point - mean_gaussian2).transpose();
//         else
//             mean_subtracted.row(i) = (data_point - mean_gaussian3).transpose();
//     }

//     VectorXd numerators(gaussians);
//     double denominator = 0;
//     VectorXd posteriors(gaussians);
//     // cout<<mean_subtracted<<endl;
//     // cout<<mean_subtracted.row(0)<<endl;
//     for (int i = 0; i < mean_subtracted.rows(); i++)
//     {
//         if (i == 0)
//         {
//             VectorXd product = ((mean_subtracted.row(i)) * (gaussian1_covariance.inverse()));
//             VectorXd exp_input = (product.transpose()) * (mean_subtracted.row(i).transpose());
//             numerators[i] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((gaussian1_covariance.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
//             numerators[i] = (numerators[i] * prior_probability[i]);
//             denominator += numerators[i];
//         }
//         if (i == 1)
//         {
//             VectorXd product = ((mean_subtracted.row(i)) * (gaussian2_covariance.inverse())).transpose();
//             VectorXd exp_input = (product.transpose()) * (mean_subtracted.row(i).transpose());
//             numerators[i] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((gaussian2_covariance.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
//             numerators[i] = numerators[i] * prior_probability[i];
//             denominator += numerators[i];
//         }
//         if (i == 2)
//         {
//             VectorXd product = ((mean_subtracted.row(i)) * (gaussian3_covariance.inverse())).transpose();
//             VectorXd exp_input = (product.transpose()) * (mean_subtracted.row(i).transpose());
//             numerators[i] = (pow((2.0 * M_PI), (-dimensions / 2)) * pow((gaussian3_covariance.determinant()), (-0.5))) * exp((-0.5) * (exp_input.value()));
//             numerators[i] = numerators[i] * prior_probability[i];
//             denominator += numerators[i];
//         }
//     }

//     posteriors = (numerators / denominator);
//     // cout<<posteriors.sum()<<endl;
//     // cout<<posteriors<<endl;
//     double max_posterior = posteriors.maxCoeff();

//     // // VectorXd assigned_gaussian(gaussians);
//     // for (int i = 0; i < gaussians; i++)
//     // {
//     //     if (posteriors[i] == max_posterior)
//     //     {

//     //     }
//     // }
//     // // cout<<assigned_gaussian<<endl;
//     // return assigned_gaussian;
//     cout << posteriors.transpose() << endl;
// }

void get_allotted_cluster()
{
    for (int i = 0; i < observation.rows(); i++)
    {
        double max_posterior = 0;
        double posterior1 = 0, posterior2 = 0, posterior3 = 0;
        posterior1 = posterior_prob_gaussian1[i];
        posterior2 = posterior_prob_gaussian2[i];
        posterior3 = posterior_prob_gaussian3[i];
        VectorXd posteriors(gaussians);
        posteriors[0] = posterior1;
        posteriors[1] = posterior2;
        posteriors[2] = posterior3;

        max_posterior = posteriors.maxCoeff();

        if (max_posterior == posterior1)
            allotted_cluster[i] = 0;
        else if (max_posterior == posterior2)
            allotted_cluster[i] = 1;
        else
            allotted_cluster[i] = 2;
    }
    cout << allotted_cluster.transpose() << endl;
}

void initialize_gaussian_parameters_before_shuffle()
{
    assign_points_to_gaussians();

    get_gaussian_initial_mean();

    get_initial_covariance();

    initialize_gaussian_priors();
}

void initialize_gaussian_parameters_after_shuffle()
{
    assign_points_to_gaussians();
    get_gaussian_initial_mean();
    get_initial_covariance();
    initialize_gaussian_priors();
}
int main()
{
    // read data from csv file
    read_csv_data("iris.txt");

    // getting initial mean and covariance for gaussians before shuffling the data

    initialize_gaussian_parameters_before_shuffle();

    // shuffling the observations
    shuffle_data();

    // getting initial mean and covariance for gaussians after shuffling the data

    // initializing the parameters of the gaussians randomly

    // initialize_gaussian_parameters_after_shuffle();
    

    for (int i = 0; i < iterations; i++)
    {
        e_step();
        if (i == 0)
        {
            prev_log_likelihood = get_log_likelihood();
            cout << "Iteration   " << i << "   Log-likelihood - " << prev_log_likelihood << endl;
        }
        else
        {
            next_log_likelihood = get_log_likelihood();
            cout << "Iteration   " << i << "   Log-likelihood - " << next_log_likelihood << endl;
        }
        m_step();
        bool stop = check_stopping_condition();

        if (stop == true)
            break;
        else
        {
            prev_log_likelihood = next_log_likelihood;
            next_log_likelihood = 0;
        }
    }
    
    cout<<actual_cluster.transpose()<<endl;
    get_allotted_cluster();

    double sum_true_allocations = 0;
    for (int i = 0; i < observation.rows(); i++)
    {
        if (allotted_cluster[i] == actual_cluster[i])
            sum_true_allocations += 1;
    }
    cout << sum_true_allocations << endl;

    return 0;
}