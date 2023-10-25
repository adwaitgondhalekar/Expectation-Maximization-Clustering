#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
vector<float> sepal_length;
vector<float> sepal_width;
vector<float> petal_length;
vector<float> petal_width;

void read_csv_data(string filename)
{
    
    fstream file;
    string line;
    file.open(filename, ios::in|ios::out);
    
    while(file)
    {
        
        getline(file, line);
        istringstream is(line);
        int params=1;
        string data="";
        while(getline(is,data,','))
        {
            switch (params)
            {
            case 1:
                sepal_length.push_back(stof(data));
                break;
            case 2:
                sepal_width.push_back(stof(data));
                break;
            case 3:
                petal_length.push_back(stof(data));
                break;
            case 4:
                petal_width.push_back(stof(data));
                break;
            default:
                break;
            }
            params++;
        }

    }

    


}

int main()
{
    read_csv_data("iris.txt");
    return 0;
}