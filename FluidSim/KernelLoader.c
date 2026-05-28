 #include <stdio.h>
 #include <stdlib.h>
 #include "KernelLoader.h"

char* loadKernelSource(const char* fileName){
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
    {
        printf("[ERROR] Kernel megnyitás sikertelen: %s\n", fileName);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char* source = (char*)malloc(fileSize + 1);
    if (source == NULL)
    {
        printf("[ERROR] Memória foglalás sikertelen");
        fclose(file);
        return NULL;
    }

    size_t bytesRead = fread(source, 1, fileSize, file);
    source[bytesRead] = '\0';

    fclose(file);
    return source;
    
    
}
