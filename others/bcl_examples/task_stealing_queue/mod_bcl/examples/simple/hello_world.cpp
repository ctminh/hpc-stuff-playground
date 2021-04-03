#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>

int main(int argc, char **argv)
{
  BCL::init();
  printf("Hello, BCL! I am rank %lu/%lu on host %s.\n",
         BCL::rank(), BCL::nprocs(), BCL::hostname().c_str());

  std::vector<BCL::CircularQueue<std::string>> queues;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++)
  {
    queues.push_back(BCL::CircularQueue<std::string>(rank, 33550));
  }

  std::string t = "Hello";
  if(BCL::rank() == 0)
    queues[0].push(t);

  BCL::barrier();

  if(BCL::rank()== 1){
    std::string s;
    queues[0].pop(s);
    std::cout << s << "\n";
  }
  BCL::finalize();
  return 0;
}
