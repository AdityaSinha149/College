#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <pwd.h>
#include <grp.h>
#include <time.h>

void perms(mode_t m){
    char p[11]="----------";
    if(S_ISDIR(m)) p[0]='d';
    if(m&S_IRUSR) p[1]='r'; if(m&S_IWUSR) p[2]='w'; if(m&S_IXUSR) p[3]='x';
    if(m&S_IRGRP) p[4]='r'; if(m&S_IWGRP) p[5]='w'; if(m&S_IXGRP) p[6]='x';
    if(m&S_IROTH) p[7]='r'; if(m&S_IWOTH) p[8]='w'; if(m&S_IXOTH) p[9]='x';
    printf("%s ",p);
}

int main(){
    DIR *d=opendir(".");
    struct dirent *e; struct stat s; struct passwd *pw; struct group *gr; char t[64];
    while((e=readdir(d))){
        if(e->d_name[0]=='.') continue;
        stat(e->d_name,&s);
        perms(s.st_mode);
        printf("%2ld %s %s %6ld ",(long)s.st_nlink,
               (pw=getpwuid(s.st_uid))?pw->pw_name:"?", 
               (gr=getgrgid(s.st_gid))?gr->gr_name:"?", 
               (long)s.st_size);
        strftime(t,sizeof(t),"%b %d %H:%M",localtime(&s.st_mtime));
        printf("%s %s\n",t,e->d_name);
    }
    closedir(d);
}
