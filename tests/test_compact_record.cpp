#include "lib/compact_training_record.h"
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace CompactTrainingRecord;
static void check(bool value, const char* text) { if (!value) { std::cerr << text << '\n'; std::exit(1); } }
static bool same(const Entry& a, const Entry& b) {
    if (a.kings != b.kings || a.turn != b.turn || a.score != b.score || a.result != b.result || a.pieces.size() != b.pieces.size()) return false;
    for (unsigned i=0;i<a.pieces.size();++i)
        if (a.pieces[i].square!=b.pieces[i].square || a.pieces[i].type!=b.pieces[i].type || a.pieces[i].color!=b.pieces[i].color) return false;
    return true;
}
int main() {
    std::string error;
    Codec codec;
    check(!codec.reset(64,5,6,error) && !codec.reset(64,5,145,error), "invalid widths accepted");
    std::array<std::uint8_t,144> bytes;
    Entry input, output;
    input.kings={0,63}; input.turn=1; input.score=-31999; input.result=-1;
    for (unsigned size=7;size<=144;++size) {
        check(codec.reset(64,5,size,error), "valid width rejected");
        for (unsigned n=0;n<=5;++n) {
            input.pieces.clear();
            for (unsigned i=0;i<n;++i) input.pieces.push_back({std::uint16_t(i+1),std::uint16_t(i%5),std::uint8_t(i%2)});
            bool fits=codec.can_fit(n);
            check(codec.encode(input,bytes.data(),error)==fits,"capacity check mismatch");
            if (fits) check(codec.decode(bytes.data(),output,error)&&same(input,output),"roundtrip mismatch");
        }
    }
    input.pieces={{12,2,0},{42,4,1}};
    check(codec.reset(64,5,8,error)&&codec.encode(input,bytes.data(),error),"8-byte encode failed");
    std::uint64_t legacy=std::uint64_t(63)<<6;
    for (unsigned i=0;i<2;++i) {
        auto p=input.pieces[i];
        legacy|=std::uint64_t(p.square|(p.type<<6)|(p.color<<9))<<(12+10*i);
    }
    legacy|=(std::uint64_t(1)<<42)|(std::uint64_t(2)<<43)|(std::uint64_t(std::uint16_t(-31999))<<45);
    for(unsigned i=0;i<8;++i) check(bytes[i]==((legacy>>(8*i))&255),"legacy 8-byte bytes changed");
    check(codec.reset(64,5,7,error)&&codec.can_fit(2)&&!codec.can_fit(3),"7-byte sparse capacity wrong");
    check(codec.encode(input,bytes.data(),error),"7-byte sparse encoding failed");
    bytes[6]|=64;
    check(!codec.decode(bytes.data(),output,error),"nonzero padding accepted");
    input.kings={0,11}; input.pieces={{1,0,0},{2,1,1},{3,2,0},{4,3,1}};
    check(codec.reset(12,5,7,error)&&codec.can_fit(4),"tiny six-piece capacity wrong");
    check(codec.encode(input,bytes.data(),error)&&(bytes[6]&128),"bitmap mode not selected");
    check(codec.decode(bytes.data(),output,error)&&same(input,output),"bitmap roundtrip failed");
    bytes[2]=std::uint8_t((bytes[2]&240)|5); // second occupied square becomes another White King
    check(!codec.decode(bytes.data(),output,error),"duplicate bitmap King accepted");
    input.kings={0,4095}; input.pieces.clear();
    for(unsigned i=0;i<40;++i) input.pieces.push_back({std::uint16_t(i+1),std::uint16_t(i%32),std::uint8_t(i%2)});
    check(codec.reset(4096,32,144,error)&&codec.encode(input,bytes.data(),error),"wide schema failed");
    check(codec.decode(bytes.data(),output,error)&&same(input,output),"wide schema roundtrip failed");
    input.pieces[1].type=299;
    check(codec.reset(4096,512,144,error)&&codec.encode(input,bytes.data(),error),"wide type encode failed");
    check(codec.decode(bytes.data(),output,error)&&same(input,output),"wide type index truncated");
    input.pieces[1].square=input.pieces[0].square;
    check(!codec.encode(input,bytes.data(),error),"overlapping supports accepted");
    std::cout<<"Adaptive compact records: all widths, legacy bytes, bitmap, malformed and wide profiles passed\n";
}
