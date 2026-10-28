/* Stream JSON while eliding numeric-only metadata arrays. Required input
 * containers and every string/object scalar are preserved. A mixed array is
 * replayed exactly in value; large undecided arrays spill to a temporary file.
 */
#include <yajl/yajl_parse.h>
#include <yajl/yajl_gen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP (8*1024*1024)
#define DEPTH 4096
typedef struct {
    yajl_gen gen;
    unsigned char *buffer;
    size_t used;
    FILE *spill;
    int capture, capture_depth, depth, pending_protected, protected[DEPTH];
    unsigned long long arrays_elided, numbers_elided, capture_numbers;
    int failed;
} state;

static void output(void *ctx,const char *text,size_t n) {
    state *s=ctx;
    if (!s->capture) { if(fwrite(text,1,n,stdout)!=n)s->failed=1; return; }
    if (!s->spill && s->used+n<=CAP) {memcpy(s->buffer+s->used,text,n);s->used+=n;return;}
    if (!s->spill) {
        s->spill=tmpfile();
        if(!s->spill){s->failed=1;return;}
        if(fwrite(s->buffer,1,s->used,s->spill)!=s->used)s->failed=1;
        s->used=0;
    }
    if(fwrite(text,1,n,s->spill)!=n)s->failed=1;
}

static void flush_capture(state *s,int keep) {
    s->capture=0;
    if(s->spill) {
        if(keep) {
            unsigned char block[65536];size_t n;
            if(fseek(s->spill,0,SEEK_SET))s->failed=1;
            while((n=fread(block,1,sizeof block,s->spill)))if(fwrite(block,1,n,stdout)!=n)s->failed=1;
            if(ferror(s->spill))s->failed=1;
        }
        fclose(s->spill);s->spill=NULL;
    } else if(keep && fwrite(s->buffer,1,s->used,stdout)!=s->used)s->failed=1;
    if(!keep){s->arrays_elided++;s->numbers_elided+=s->capture_numbers;}
    s->used=0;s->capture_numbers=0;
}

static int ok(state *s,yajl_gen_status status){return !s->failed && status==yajl_gen_status_ok;}
static void scalar(state *s){s->pending_protected=0;}
static int null_value(void *ctx){state *s=ctx;scalar(s);return ok(s,yajl_gen_null(s->gen));}
static int boolean_value(void *ctx,int value){state *s=ctx;if(s->capture)flush_capture(s,1);scalar(s);return ok(s,yajl_gen_bool(s->gen,value));}
static int number_value(void *ctx,const char *value,size_t n){state *s=ctx;if(s->capture)s->capture_numbers++;scalar(s);return ok(s,yajl_gen_number(s->gen,value,n));}
static int string_value(void *ctx,const unsigned char *value,size_t n){state *s=ctx;if(s->capture)flush_capture(s,1);scalar(s);return ok(s,yajl_gen_string(s->gen,value,n));}
static int push(state *s){if(s->depth+1>=DEPTH)return 0;s->protected[s->depth+1]=s->protected[s->depth]||s->pending_protected;s->pending_protected=0;s->depth++;return 1;}
static int map_start(void *ctx){state *s=ctx;if(s->capture)flush_capture(s,1);return push(s)&&ok(s,yajl_gen_map_open(s->gen));}
static int map_end(void *ctx){state *s=ctx;if(s->depth<1)return 0;s->depth--;return ok(s,yajl_gen_map_close(s->gen));}
static int map_key(void *ctx,const unsigned char *value,size_t n){
    state *s=ctx;
    const char *keys[]={"child_section_ids","spans","fragments","messages","support","turns",NULL};
    s->pending_protected=0;
    for(int i=0;keys[i];i++)if(strlen(keys[i])==n&&!memcmp(value,keys[i],n))s->pending_protected=1;
    return ok(s,yajl_gen_string(s->gen,value,n));
}
static int array_start(void *ctx){
    state *s=ctx;if(!push(s))return 0;
    if(!ok(s,yajl_gen_array_open(s->gen)))return 0;
    if(!s->capture && !s->protected[s->depth]){s->capture=1;s->capture_depth=s->depth;s->used=0;s->capture_numbers=0;}
    return 1;
}
static int array_end(void *ctx){
    state *s=ctx;if(s->depth<1)return 0;
    if(s->capture && s->capture_depth==s->depth)flush_capture(s,0);
    s->depth--;return ok(s,yajl_gen_array_close(s->gen));
}

int main(void){
    state s={0};s.buffer=malloc(CAP);if(!s.buffer)return 2;
    s.gen=yajl_gen_alloc(NULL);if(!s.gen)return 2;
    yajl_gen_config(s.gen,yajl_gen_print_callback,output,&s);
    yajl_callbacks callbacks={null_value,boolean_value,NULL,NULL,number_value,string_value,map_start,map_key,map_end,array_start,array_end};
    yajl_handle parser=yajl_alloc(&callbacks,NULL,&s);if(!parser)return 2;
    unsigned char input[65536];size_t n;yajl_status status=yajl_status_ok;
    while((n=fread(input,1,sizeof input,stdin))){status=yajl_parse(parser,input,n);if(status!=yajl_status_ok)break;}
    if(status==yajl_status_ok)status=yajl_complete_parse(parser);
    int result=status==yajl_status_ok && !s.failed && !ferror(stdin) && !fflush(stdout)?0:1;
    fprintf(stderr,"{\"numeric_metadata_arrays_elided\":%llu,\"number_values_elided\":%llu,\"exit_code\":%d}\n",s.arrays_elided,s.numbers_elided,result);
    if(s.spill)fclose(s.spill);yajl_free(parser);yajl_gen_free(s.gen);free(s.buffer);return result;
}
