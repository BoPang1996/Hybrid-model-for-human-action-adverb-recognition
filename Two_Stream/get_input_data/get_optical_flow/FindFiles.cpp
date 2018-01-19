#include "FindFiles.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>  
#include <stdio.h> 

#ifdef WIN32

#include <Windows.h>
#include <direct.h>

std::vector<std::string> FindFiles::findFiles( const char *lpPath, const char *secName /*= ".*" */ )
{
    char szFind[MAX_PATH];
    char szFile[MAX_PATH];

    WIN32_FIND_DATA FindFileData;

    strcpy(szFind,lpPath);
    strcat(szFind,"\\*");
    strcat(szFind,secName);

    HANDLE hFind=::FindFirstFile(szFind,&FindFileData);

    if(INVALID_HANDLE_VALUE == hFind)
    {
        std::cout << "Empty folder!" << std::endl;
        return std::vector<std::string>();
    }

    do
    {
        if(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            if(FindFileData.cFileName[0]!='.')
            {
                strcpy(szFile,lpPath);
                strcat(szFile,"\\");
                strcat(szFile,FindFileData.cFileName);
                findFiles(szFile);
            }
        }
        else
        {
            if ( szFile[0] )
            {
                std::string filePath = lpPath;
                filePath += "\\";
                filePath += FindFileData.cFileName;
                file_lists.push_back(filePath);
            }
            else
            {
                std::string filePath = szFile;
                filePath += FindFileData.cFileName;
                file_lists.push_back(filePath);
            }
        }

    }while(::FindNextFile(hFind,&FindFileData));

    ::FindClose(hFind);
    return file_lists;
}

#else


#include <string.h>
#include <queue>
#include "dirent.h"

std::vector<std::string> FindFiles::findFiles( const char *lpPath, const char *secName /*= ".*" */ )
{
    (void)secName;

    std::vector<std::string> result;
    std::queue<std::string> queue;
    std::string dirname;

    DIR *dir;
    if ( !(dir = opendir ( lpPath )) ) {
        return result;
    }
    queue.push( lpPath );

    struct dirent *ent;
    while ( !queue.empty() )
    {

        dirname = queue.front();
        dir = opendir( dirname.c_str() );
        queue.pop();
        if ( !dir ) { continue; }

        while( ent = readdir( dir ) )
        {
            if ( strcmp(".", ent->d_name) == 0 || strcmp("..", ent->d_name) == 0 )
            {
                continue;
            }
            if ( ent->d_type == DT_DIR )
            {
                queue.push( dirname+"/"+ent->d_name );
            }
            else
            {
                result.push_back( dirname+"/"+ent->d_name );
            }

        }

        closedir( dir );

    }

    return result;
}

#endif
