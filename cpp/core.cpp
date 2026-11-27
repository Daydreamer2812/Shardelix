// Portable C++20 kernels. The ABI transfers flat arrays; no model or ground truth.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <array>
#include <new>
#include <tuple>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
#define FL_EXPORT __declspec(dllexport)
#else
#define FL_EXPORT __attribute__((visibility("default")))
#endif
namespace {
int mod4(int x) { return x & 3; }
int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};
}
extern "C" {
FL_EXPORT int fl_abi() {return 1;}
FL_EXPORT void fl_free(void* p) {std::free(p);}
// Exact clockwise vertex contours, including holes and diagonal contacts.
// Reproduce the reference insertion order (N,E,S,W; row major).
FL_EXPORT int32_t* fl_contours(const uint8_t* mask,int h,int w) {
    if(!mask||h<=0||w<=0||int64_t(h)*w>110000000) return nullptr;
    const int stride=w+1;
    std::unordered_map<int,uint8_t> edges;
    std::vector<int> order;
    for(int d=0;d<4;++d) for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        if(!mask[y*w+x]) continue;
        int nx=x+dy[d],ny=y-dx[d];
        if(nx>=0&&nx<w&&ny>=0&&ny<h&&mask[ny*w+nx]) continue;
        int key=(y+(d>=2))*stride+x+((d==1||d==2)?1:0);
        auto [it,inserted]=edges.try_emplace(key,0);
        if(inserted) order.push_back(key);
        it->second|=uint8_t(1u<<d);
    }
    std::vector<int32_t> out{0};
    std::vector<int32_t> contour;
    for(int start:order) while(edges.contains(start)) {
        int p=start,prev=-1; contour.clear();
        while(true) {
            auto it=edges.find(p);if(it==edges.end())break;
            uint8_t bits=it->second;int d=-1;
            if(prev<0){for(int k=0;k<4;++k)if(bits&(1u<<k)){d=k;break;}}
            else {for(int k:{mod4(prev+1),prev,mod4(prev-1),mod4(prev+2)})if(bits&(1u<<k)){d=k;break;}}
            if(d<0)break;
            it->second=uint8_t(bits&~(1u<<d)); if(!it->second)edges.erase(it);
            contour.insert(contour.end(),{p%stride,p/stride,d});
            p+=dy[d]*stride+dx[d];prev=d;
            if(p==start)break;
        }
        if(p==start&&contour.size()>=12){out.push_back(int32_t(contour.size()/3));out.insert(out.end(),contour.begin(),contour.end());}
    }
    out[0]=int32_t(out.size());
    auto* p=static_cast<int32_t*>(std::malloc(out.size()*sizeof(int32_t)));
    if(p)std::copy(out.begin(),out.end(),p);return p;
}
// Exact complementary chain-code voting. Each 2-bit direction difference
// is lossless under quarter-turns. Canonical hash shards retain BOTH sides of
// every possible match, so a smaller index budget changes memory, not recall.
FL_EXPORT int32_t* fl_arc_votes(const int32_t* packed,int size,int k,int bucket,
                               int shards,int record_cap,int vote_cap) {
    if(!packed||size<2||k<6||k>32||bucket<1||shards<1||record_cap<1||vote_cap<1)return nullptr;
    struct Loop{int i,n;const int32_t* p;};
    struct Arc{uint64_t code,rev;int i,x,y,bx,by;uint8_t da,db;};
    using Vote=std::array<int32_t,5>;
    struct Hash{size_t operator()(const Vote& q)const{
        uint64_t h=0x9e3779b97f4a7c15ULL;
        for(auto v:q)h^=uint32_t(v)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        return size_t(h);
    }};
    try {
        std::vector<Loop> loops;
        for(int p=0;p<size;){
            if(size-p<2)return nullptr;
            const int i=packed[p++],n=packed[p++];
            if(i<0||n<4||int64_t(n)*3>size-p)return nullptr;
            if(n>k)loops.push_back({i,n,packed+p});
            p+=3*n;
        }
        const int m=k-1,shift=2*(m-1);
        const uint64_t mask=(uint64_t(1)<<(2*m))-1;
        std::unordered_map<Vote,int32_t,Hash> votes;
        auto mix=[](uint64_t q){q^=q>>30;q*=0xbf58476d1ce4e5b9ULL;q^=q>>27;q*=0x94d049bb133111ebULL;return q^(q>>31);};
        for(int shard=0;shard<shards;++shard){
            std::vector<Arc> arcs;
            // A truly allocated bound prevents an underestimated shard from
            // turning into an unbounded memory allocation.
            arcs.reserve(std::min(record_cap, std::max(1024,size/3/shards)));
            for(const auto& l:loops){
                auto d=[&](int z){return l.p[3*(z%l.n)+2];};
                uint64_t f=0,r=0;int changes=0;
                auto turn=[&](int z){return mod4(d(z+1)-d(z));};
                for(int z=0;z<m;++z){int v=turn(z);f=(f<<2)|v;r=(r>>2)|(uint64_t(mod4(-v))<<shift);changes+=v!=0;}
                for(int s=0;s<l.n;++s){
                    if(changes>=4&&int(mix(std::min(f,r))%uint64_t(shards))==shard){
                        if(int(arcs.size())>=record_cap)return nullptr;
                        if(arcs.size()==arcs.capacity())arcs.reserve(std::min<size_t>(record_cap,std::max<size_t>(1024,arcs.capacity()*2)));
                        const int a=3*s,b=3*((s+k)%l.n);
                        arcs.push_back({f,r,l.i,l.p[a],l.p[a+1],l.p[b],l.p[b+1],uint8_t(d(s)),uint8_t(mod4(d(s+k-1)+2))});
                    }
                    int old=turn(s),next=turn(s+m);
                    changes+=(next!=0)-(old!=0);f=((f<<2)|next)&mask;
                    r=(r>>2)|(uint64_t(mod4(-next))<<shift);
                }
            }
            std::sort(arcs.begin(),arcs.end(),[](const Arc& a,const Arc& b){return a.code<b.code;});
            for(size_t a=0;a<arcs.size();){
                size_t b=a+1;while(b<arcs.size()&&arcs[b].code==arcs[a].code)++b;
                if(b-a<=size_t(bucket)){
                    // For a fixed normalized direction chain, the reversed
                    // complementary chain is unique.
                    auto it=std::lower_bound(arcs.begin(),arcs.end(),arcs[a].rev,[](const Arc& q,uint64_t v){return q.code<v;});
                    auto end=it;while(end!=arcs.end()&&end->code==arcs[a].rev&&end-it<=bucket)++end;
                    if(end-it<=bucket)for(size_t z=a;z<b;++z)for(auto jt=it;jt!=end;++jt){
                        const Arc&A=arcs[z];const Arc&B=*jt;if(A.i>=B.i)continue;
                        int rr=mod4(A.da-B.db),x=B.bx,y=B.by;
                        int xx=rr==0?x:rr==1?-y:rr==2?-x:y;
                        int yy=rr==0?y:rr==1?x:rr==2?-y:-x;
                        Vote v{A.i,B.i,rr,A.x-xx,A.y-yy};
                        auto [where,fresh]=votes.try_emplace(v,0);
                        if(fresh&&int(votes.size())>vote_cap)return nullptr;
                        ++where->second;
                    }
                }
                a=b;
            }
        }
        std::vector<std::array<int32_t,6>> result;
        for(const auto& [q,v]:votes)if(v>=3)result.push_back({v,q[0],q[1],q[2],q[3],q[4]});
        std::sort(result.begin(),result.end(),[](const auto&a,const auto&b){return a[0]!=b[0]?a[0]>b[0]:a<b;});
        int64_t n=1+6*int64_t(result.size());if(n>std::numeric_limits<int32_t>::max())return nullptr;
        auto* out=static_cast<int32_t*>(std::malloc(size_t(n)*sizeof(int32_t)));if(!out)return nullptr;
        out[0]=int32_t(n);int32_t* p=out+1;
        for(const auto&q:result){std::copy(q.begin(),q.end(),p);p+=6;}
        return out;
    }catch(const std::bad_alloc&){return nullptr;}
}
// Exact integer raster adjacency and overlap. 766-bin RGB absolute-colour
// histogram gives exact median and linear 80th percentile without copies.
FL_EXPORT int fl_lattice_contact(const uint8_t* a,int h,int w,const uint8_t* b,int bh,int bw,
                                 int r,int tx,int ty,double* out) {
    if(!a||!b||!out||h<=0||w<=0||bh<=0||bw<=0||r<0||r>3)return 0;
    const int X=tx-(r==1?bh:r==2?bw:0),Y=ty-(r==2?bh:r==3?bw:0);
    const int W=r%2?bh:bw,H=r%2?bw:bh;
    auto get=[&](int x,int y){
        int u=r==0?x:r==1?y:r==2?bw-1-x:bw-1-y;
        int v=r==0?y:r==1?bh-1-x:r==2?bh-1-y:x;
        return b+4*(int64_t(v)*bw+u);
    };
    auto visit=[&](int x,int y,auto fn){
        for(int y0=std::max(0,y);y0<std::min(h,y+H);++y0)
            for(int x0=std::max(0,x);x0<std::min(w,x+W);++x0){
                const uint8_t* p=a+4*(int64_t(y0)*w+x0);
                if(!p[3])continue;
                const uint8_t* q=get(x0-x,y0-y);
                if(q[3]&&!fn(p,q))return false;
            }
        return true;
    };
    if(!visit(X,Y,[](auto,auto){return false;}))return 0;
    std::array<int64_t,766> hist{};int64_t length=0;
    auto add=[&](const uint8_t*p,const uint8_t*q){int error=0;for(int c=0;c<3;++c)error+=std::abs(int(p[c])-q[c]);++hist[error];++length;return true;};
    for(int d=0;d<4;++d)visit(X+dx[d],Y+dy[d],add);
    if(length<12)return 0;
    auto value=[&](int64_t rank){int64_t cum=0;for(int z=0;z<766;++z){cum+=hist[z];if(cum>rank)return z;}return 765;};
    auto quantile=[&](double q){double pos=q*(length-1);int64_t i=int64_t(pos);double f=pos-i;return ((1-f)*value(i)+f*value(std::min(i+1,length-1)))/3.;};
    out[0]=double(length);out[1]=quantile(.5);out[2]=quantile(.8);return 1;
}
// Batch edge descriptors: only boundary and inward pixels, not whole RGB images.
FL_EXPORT int fl_profiles(const uint8_t* rgba,int n,int h,int w,float* out) {
    if(!rgba||!out||n<1||h<1||w<1)return 0;
    const int total=2*(h+w);
    auto color=[](const uint8_t* p,float* v){
        float r=p[0]/255.f,g=p[1]/255.f,b=p[2]/255.f;
        v[0]=.299f*r+.587f*g+.114f*b;v[1]=(r-g)*.65f;v[2]=(b-g)*.65f;
    };
    for(int i=0;i<n;++i){
        const uint8_t* a=rgba+int64_t(i)*h*w*4;
        int offset=0;
        for(int side=0;side<4;++side){
            int L=side%2==0?w:h;
            for(int k=0;k<L;++k){
                int x,y,X,Y;
                if(side==0){x=k;y=0;X=x;Y=std::min(1,h-1);}
                else if(side==1){x=w-1;y=k;X=std::max(0,w-2);Y=y;}
                else if(side==2){x=w-1-k;y=h-1;X=x;Y=std::max(0,h-2);}
                else{x=0;y=h-1-k;X=std::min(1,w-1);Y=y;}
                float e[3],q[3];color(a+4*(y*w+x),e);color(a+4*(Y*w+X),q);
                float* v=out+(int64_t(i)*total+offset+k)*6;
                for(int c=0;c<3;++c){v[c]=e[c];v[c+3]=e[c]-q[c];}
            }
            offset+=L;
        }
    }
    return 1;
}
// Exact reciprocal 2x2 cycle support. O(edges*keep), no dense matrix.
FL_EXPORT int fl_loop_votes(const int32_t* idx,int m,int keep,int32_t* out) {
    if(!idx||!out||m<=0||m%4||keep<=0||keep>256)return 0;
    const int top=std::min(2,keep);
    std::fill(out,out+int64_t(m)*keep,0);
    auto get=[&](int a,int r){return idx[int64_t(a)*keep+r];};
    auto valid=[&](int a){return a>=0&&a<m;};
    auto add=[&](int a,int b){for(int t=0;t<keep;++t){if(get(a,t)==b)++out[int64_t(a)*keep+t];if(get(b,t)==a)++out[int64_t(b)*keep+t];}};
    for(int a=0;a<m;++a){
        const int i=a/4,s=a%4,c=4*i+mod4(s+1);
        for(int rb=0;rb<top;++rb){
            int b=get(a,rb);if(!valid(b))continue;
            int j=b/4,t=b%4,js=4*j+mod4(s+1-mod4(s+2-t));
            for(int re=0;re<top;++re){
                int e=get(c,re);if(!valid(e)||e/4==j)continue;
                int l=e/4,u=e%4,ls=4*l+mod4(s-mod4(s+3-u));
                for(int rf=0;rf<top;++rf){
                    int f=get(js,rf);if(!valid(f))continue;
                    int z=f/4,v=f%4;if(z==i||z==j||z==l)continue;
                    int g=4*z+mod4(s+2-mod4(s+3-v));
                    bool ok=false;for(int r=0;r<top;++r)if(get(ls,r)==g)ok=true;
                    if(ok){add(a,b);add(c,e);add(js,f);add(ls,g);}
                }
            }
        }
    }
    return 1;
}
// Exact sparse seam score: fused passes avoid materializing candidate-by-pixel tensors.
// Inputs are flattened RGB triples; candidate indices local to target descriptors.
FL_EXPORT int fl_rerank(const float* e,const float* g,const float* f,const float* h,
  const int32_t* jj,int rows,int targets,int k,int length,int step,float* out) {
    if(!e||!g||!f||!h||!jj||!out||rows<1||targets<1||k<1||length<1||step<1)return 0;
    const int samples=(length+step-1)/step;
    for(int i=0;i<rows;++i) {
        const float* E=e+int64_t(i)*length*3,*G=g+int64_t(i)*length*3;
        float gs[3]={};for(int l=0;l<length;l+=step)for(int c=0;c<3;++c)gs[c]+=G[3*l+c]*G[3*l+c];
        float su[3];for(int c=0;c<3;++c)su[c]=std::sqrt(gs[c]/samples+.00012f);
        float gu=std::max(.01f,std::sqrt((gs[0]+gs[1]+gs[2])/(samples*3)));
        for(int r=0;r<k;++r){
            int j=jj[int64_t(i)*k+r];if(j<0||j>=targets)return 0;
            const float *F=f+int64_t(j)*length*3,*H=h+int64_t(j)*length*3;
            float base=0,sg[3]={},sh[3]={},hs[3]={};
            for(int l=0;l<length;l+=step){
                int v=length-1-l;
                for(int c=0;c<3;++c){
                    float delta=E[3*l+c]-F[3*v+c],hv=H[3*v+c];
                    base+=std::abs(delta);sg[c]+=std::abs(delta+G[3*l+c]);sh[c]+=std::abs(-delta+hv);hs[c]+=hv*hv;
                }
            }
            float hv=std::max(.01f,std::sqrt((hs[0]+hs[1]+hs[2])/(samples*3)));
            float pred=0;for(int c=0;c<3;++c)pred+=(sg[c]/samples)/su[c]+(sh[c]/samples)/std::sqrt(hs[c]/samples+.00012f);
            out[int64_t(i)*k+r]=.45f*(base/(samples*3))/std::sqrt(gu*hv)+.55f*pred/6.f;
        }
    }
    return 1;
}
}
