namespace MiniGrad
{
    template <typename T>
    class TensorStorage
    {
    public:
        TensorStorage(int size)
        {
            m_data = new T[size];
        }

        ~TensorStorage() 
        {
            delete[] m_data;
            m_data = nullptr;
        }

    private:
        T* m_data;
    };
}
