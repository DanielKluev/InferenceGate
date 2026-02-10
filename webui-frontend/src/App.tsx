import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CacheList from './pages/CacheList';
import EntryDetail from './pages/EntryDetail';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/cache" element={<CacheList />} />
          <Route path="/cache/:id" element={<EntryDetail />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;

