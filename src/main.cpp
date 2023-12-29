#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <omp.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <data_prefix>\n";
        return 1;
    }
    std::string data_prefix = argv[1];
    // file_prefix: "data:"
    std::string ref_path = data_prefix + "/input" + "/ref.txt";
    std::cout << ref_path << std::endl;
    std::ifstream ref(ref_path);

    if (!ref.is_open())
    {
        std::cout << "Error opening ref.txt\n";
        return 1;
    }

    std::string line;
    int filecount = 0;
    std::vector<std::string> input_paths;
    while (std::getline(ref, line))
    {
        std::string input_path = data_prefix + "/input/" + std::to_string(filecount) + ".txt";
        std::ofstream input(input_path);
        input_paths.push_back(input_path);

        input << line << "\n";
        std::getline(ref, line);
        input << line << "\n";
        filecount++;
        input.close();
    }

    std::string executable = "./deepsjeng";
    std::vector<std::string> arguments;

    for(auto path: input_paths)
        arguments.push_back(path);

    const int numExecutables = arguments.size();
    std::cout << "Executing " << numExecutables << " executables.\n";

#pragma omp parallel for
    for (int i = 0; i < numExecutables; ++i)
    {
        std::string command = std::string(executable) + " " + std::string(arguments[i]);
        std::cout << "Executing " << command << "\n";
        int result = std::system(command.c_str());

#pragma omp critical
        {
            if (result == 0)
            {
                std::cout << "Execution of " << command << " successful.\n";
            }
            else
            {
                std::cout << "Error executing " << command << ". Exit code: " << result << "\n";
            }
        }
    }

    // clean up the input files
    for (auto path : input_paths)
    {
        std::remove(path.c_str());
    }

    return 0;
}
