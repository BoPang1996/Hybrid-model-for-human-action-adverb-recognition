/*
 *   Find and generate a file list of the folder.
**/

#ifndef FIND_FILES_H
#define FIND_FILES_H

#include <vector>
#include <string>
using namespace std;

class FindFiles
{
public:
    FindFiles(){}
    ~FindFiles(){}

    std::vector<std::string> findFiles( const char *lpPath, const char *secName = ".*" );

private:
    std::vector<std::string> file_lists;
};

#endif // FIND_FILES_H
