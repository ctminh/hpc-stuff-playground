#include <iostream>
#include <ostream>
#include <vector>
#include <mpi.h>

struct Evento{
    char *evento;
    unsigned char cant;
};

struct Traza{
    char *nombre;
    Evento *eventos;
    unsigned int cantEventos;
    bool revisado;
    unsigned int idTraza;
};

void init_test_data(int &count, Traza * &data){
    count = 2;
    data = new Traza[2] {
        {   // the first element
            new char[3] {char(0), char(1), char(3)},    // for char *nombre
            new Evento[3] { // for Evento *eventos
                { new char[4] {'a', 'b', 'c', 'd'}, (unsigned char) 4, }, // for char *evento and unsigned char cant in struct Evento
                { new char[3] {'e', 'f', 'g'}, (unsigned char) 3, },
                { new char[2] {'h', 'i'}, (unsigned char) 2, },
            },
            3u, // for cantEventos
            true,   // for revisado
            0u, // for idTraza
        },
        {   // the second element
            new char[1] { char(4) },
            new Evento[1] {
                {new char[1] {'j'}, (unsigned char) 1, },
            },
            1u,
            false,
            1u,
        },
    };
}

void print_data(std::ostream &out, int count, const Traza *data)
{
    for(int t = 0; t < count; ++t)
    {
    std::cout << "{\n"
              << "  nombre = { ";
    for(unsigned e = 0; e < data[t].cantEventos; ++e)
      std::cout << int(data[t].nombre[e]) << ", ";
    std::cout << "},\n"
              << "  eventos = {\n";
    for(unsigned e = 0; e < data[t].cantEventos; ++e)
    {
      std::cout << "    {\n"
                << "      evento = { ";
      for(int c = 0; c < data[t].eventos[e].cant; ++c)
        std::cout << "'" << data[t].eventos[e].evento[c] << "', ";
      std::cout << "},\n"
                << "      cant = " << int(data[t].eventos[e].cant) << ",\n"
                << "    },\n";
    }
    std::cout << "  },\n"
              << "  cantEventos = " << data[t].cantEventos << ",\n"
              << "  revisado = " << data[t].revisado << ",\n"
              << "  idTraza = " << data[t].idTraza << ",\n"
              << "}," << std::endl;
  }
}

void pack_plista(int incount, Traza* data, MPI_Comm comm, std::vector<char> &buf)
{
    int pos = 0;
    buf.clear();
    int size;
    
    // return an upper bound on the amount of space needed to pack a message
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    // pack the first element is incount, type int
    MPI_Pack(&incount, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack all elements in the struct
    for (int i = 0; i < incount; ++i){
        // pack 2 fields unsiged int first: cantEventos, idTraza
        MPI_Pack_size(2, MPI_UNSIGNED, comm, &size);
        buf.resize(pos + size);
        MPI_Pack(&data[i].cantEventos, 1, MPI_UNSIGNED, buf.data(), buf.size(), &pos, comm);    // for cantEventos
        MPI_Pack(&data[i].idTraza, 1, MPI_UNSIGNED, buf.data(), buf.size(), &pos, comm);    // for idTraza

        // pack bool revisado & MPI doesn't know Bool
        MPI_Pack_size(1, MPI_UNSIGNED_CHAR, comm, &size);
        buf.resize(pos + size);
        {
            unsigned char revisado = data[i].revisado;
            MPI_Pack(&revisado, 1, MPI_UNSIGNED_CHAR, buf.data(), buf.size(), &pos, comm);
        }

        // pack *nombre, type char - so we need to know how many elements here
        MPI_Pack_size(data[i].cantEventos, MPI_CHAR, comm, &size);  // num of elements is based on cantEventos
        buf.resize(pos + size);
        MPI_Pack(data[i].nombre, data[i].cantEventos, MPI_CHAR, buf.data(), buf.size(), &pos, comm);

        // pack *eventos, type is the struct Evento
        for (int j = 0; j < data[i].cantEventos; ++j){
            // pack unsigned char cant
            MPI_Pack_size(1, MPI_UNSIGNED_CHAR, comm, &size);
            buf.resize(pos + size);
            MPI_Pack(&data[i].eventos[j].cant, 1, MPI_UNSIGNED_CHAR, buf.data(), buf.size(), &pos, comm);

            // pack char *evento
            MPI_Pack_size(data[i].eventos[j].cant, MPI_CHAR, comm, &size);
            buf.resize(pos + size);
            MPI_Pack(&data[i].eventos[j].evento, data[i].eventos[j].cant, MPI_CHAR, buf.data(), buf.size(), &pos, comm);
        }
    }
    buf.resize(pos);
    // printf("pos = %d, incount = %d, buf.size() = %ld\n", pos, incount, buf.size());
}

void unpack_plista(int &outcount, Traza* &data, MPI_Comm comm, std::vector<char> &buf)
{
    int pos = 0;    // unpack the fist is incount and save to outcount
    MPI_Unpack(buf.data(), buf.size(), &pos, &outcount, 1, MPI_INT, comm);
    data = new Traza[outcount];

    for (int i = 0; i < outcount; i++){
        // unpack 2 fields unsiged int first: cantEventos, idTraza
        MPI_Unpack(buf.data(), buf.size(), &pos, &data[i].cantEventos, 1, MPI_UNSIGNED, comm);
        MPI_Unpack(buf.data(), buf.size(), &pos, &data[i].idTraza, 1, MPI_UNSIGNED, comm);

        // unpack bool revisado & MPI doesn't know Bool
        {
            unsigned char revisado;
            MPI_Unpack(buf.data(), buf.size(), &pos, &revisado, 1, MPI_UNSIGNED_CHAR, comm);
            data[i].revisado = revisado;
        }
        
        // unpack *nombre, type char
        data[i].nombre = new char[data[i].cantEventos];
        MPI_Unpack(buf.data(), buf.size(), &pos, data[i].nombre, data[i].cantEventos, MPI_CHAR, comm);

        // unpack *eventos. type is the struct Evento
        data[i].eventos = new Evento[data[i].cantEventos];
        for (int j = 0; j < data[i].cantEventos; j++){
            // unpack unsigned char cant
            MPI_Unpack(buf.data(), buf.size(), &pos, &data[i].eventos[j].cant, 1, MPI_UNSIGNED_CHAR, comm);

            // unpack char *evento
            data[i].eventos[j].evento = new char[data[i].eventos[j].cant];
            MPI_Unpack(buf.data(), buf.size(), &pos, data[i].eventos[j].evento, data[i].eventos[j].cant, MPI_CHAR, comm);
        }
    }
    // printf("pos = %d, outcound = %d, buf.size() = %ld\n", pos, outcount, buf.size());
}

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Traza *plista = nullptr;  // create a list of Traza
    int count = 0;
   
    if(rank == 1) {
        std::vector<char> buf;
        init_test_data(count, plista);  // init data for testing
        print_data(std::cout, count, plista);
        pack_plista(count, plista, MPI_COMM_WORLD, buf);
        MPI_Send(buf.data(), buf.size(), MPI_PACKED, 0, 0, MPI_COMM_WORLD);
        // send_plista(count, plista, 0, 0, MPI_COMM_WORLD);
    } else {
        // recv_plista(count, plista, 1, 0, MPI_COMM_WORLD);
        MPI_Status status;
        MPI_Probe(1, 0, MPI_COMM_WORLD, &status);
        int size;
        MPI_Get_count(&status, MPI_PACKED, &size);
        std::vector<char> buf(size);
        printf("From R%d: size after get: size = %d\n", rank, size);
        MPI_Recv(buf.data(), buf.size(), MPI_PACKED, 1, 0, MPI_COMM_WORLD, &status);
        unpack_plista(count, plista, MPI_COMM_WORLD, buf);
        print_data(std::cout, count, plista);
    }
    
    MPI_Finalize();
}