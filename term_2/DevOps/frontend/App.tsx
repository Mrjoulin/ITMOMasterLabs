import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import TasksPage from './pages/Tasks';
import NotesPage from './pages/Notes';
import AreaDetails from './pages/AreaDetails';
import Login from './pages/Login';
import SignUp from './pages/SignUp';
import TaskModal from './components/TaskModal';
import NoteModal from './components/NoteModal';
import { api } from './services/api';
import { AppData, Task, Note, Area, Priority } from './types';

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('focusflow_token'));
  const [authPage, setAuthPage] = useState('login');
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [appData, setAppData] = useState<AppData | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Modal States
  const [editingTask, setEditingTask] = useState<Task | null>(null);
  const [editingNote, setEditingNote] = useState<Note | null>(null);

  // Load initial data on mount (if authenticated, or after login)
  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [user, areas, tasks, notes] = await Promise.all([
        api.getUser(),
        api.getAreas(),
        api.getTasks(),
        api.getNotes(),
      ]);
      setAppData({ user, areas, tasks, notes });
    } catch (error) {
      console.error("Failed to load data", error);
      // If token is invalid, logout
      if (error.message.includes('401')) {
          handleLogout();
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
        fetchData();
    }
  }, [isAuthenticated]);

  // Listen for auth:logout events from API calls (e.g., 401 unauthorized)
  useEffect(() => {
    const handleLogoutEvent = () => {
      handleLogout();
    };
    window.addEventListener('auth:logout', handleLogoutEvent);
    return () => window.removeEventListener('auth:logout', handleLogoutEvent);
  }, []);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
      localStorage.removeItem('focusflow_token');
      setIsAuthenticated(false);
      setAppData(null);
      setCurrentPage('dashboard');
  };

  const handleCreateArea = async (name: string, color: string) => {
      await api.createArea(name, color);
      fetchData();
  };

  const handleCreateTask = (areaId: number | null = null) => {
      setEditingTask({
          id: '', // Empty ID signals creation
          title: '',
          description: '',
          area_id: areaId || appData?.areas[0]?.id || null,
          priority: Priority.Medium,
          completed: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
      });
  };

  const handleCreateNote = (areaId: number | null = null) => {
      setEditingNote({
          title: '',
          content: '',
          area_id: areaId || appData?.areas[0]?.id || null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
      });
  };

  const handleSaveTask = async (task: Task) => {
      const taskToSave = {
          title: task.title,
          description: task.description,
          area_id: task.area_id,
          priority: task.priority,
          completed: task.completed,
          due_date: task.due_date,
      };
      if (task.id === '') {
          await api.createTask(taskToSave);
      } else {
          await api.updateTask(task.id, taskToSave);
      }
      fetchData();
  };

  const handleSaveNote = async (note: Note) => {
      const noteData = {
          title: note.title,
          content: note.content,
          area_id: note.area_id,
      };
      if (note.id == null || note.id === '') { // Check if id is falsy (empty string or 0 for new notes)
          await api.createNote(noteData);
      } else {
          await api.updateNote(note.id, noteData);
      }
      fetchData();
  };

  const handleToggleTask = async (taskId: string, completed: boolean) => {
      await api.toggleTaskCompletion(taskId, completed);
      fetchData();
  };




  // Simple Router
  const renderPage = () => {
    if (isLoading && !appData) {
        return (
            <div className="flex h-screen items-center justify-center bg-background-light dark:bg-background-dark">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
        );
    }
    
    const getAreaName = (areaId: number) => appData?.areas.find(a => a.id === areaId)?.name;
    const getAreaColor = (areaId: number) => appData?.areas.find(a => a.id === areaId)?.color;

    if (currentPage.startsWith('area:')) {
        const areaId = parseInt(currentPage.split(':')[1], 10);
        return (
            <AreaDetails 
                area={getAreaName(areaId) || 'Area'}
                color={getAreaColor(areaId) || 'bg-slate-500'}
                tasks={appData?.tasks.filter(t => t.area_id === areaId) || []}
                notes={appData?.notes.filter(n => n.area_id === areaId) || []}
                onOpenTask={setEditingTask}
                onOpenNote={setEditingNote}
                onToggleTask={handleToggleTask}
                onNewTask={() => handleCreateTask(areaId)}
                onNewNote={() => handleCreateNote(areaId)}
            />
        );
    }

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard 
            data={appData} 
            onNavigate={setCurrentPage}
            onNewTask={handleCreateTask}
            onNewNote={handleCreateNote}
            onOpenTask={setEditingTask}
            onOpenNote={setEditingNote}
        />;
      case 'tasks':
        return <TasksPage 
            data={appData} 
            onDataUpdate={fetchData} 
            onOpenTask={setEditingTask}
            onNewTask={handleCreateTask}
            onOpenNote={setEditingNote}
        />;
      case 'notes':
        return <NotesPage 
            data={appData} 
            onOpenNote={setEditingNote}
            onNewNote={handleCreateNote}
            onOpenTask={setEditingTask}
        />;
      default:
        return <Dashboard 
            data={appData} 
            onNavigate={setCurrentPage}
            onNewTask={handleCreateTask}
            onNewNote={handleCreateNote}
            onOpenTask={setEditingTask}
            onOpenNote={setEditingNote}
        />;
    }
  };

  const activeTasksCount = appData?.tasks.filter(t => !t.completed).length || 0;

  if (!isAuthenticated) {
    if (authPage === 'login') {
      return <Login onLogin={handleLogin} onSwitchToSignUp={() => setAuthPage('signup')} />;
    }
    return <SignUp onSwitchToLogin={() => setAuthPage('login')} onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background-light dark:bg-background-dark text-slate-900 dark:text-white font-display">
      <Sidebar 
        currentPage={currentPage} 
        onNavigate={setCurrentPage} 
        user={appData?.user || null} 
        onLogout={handleLogout}
        areas={appData?.areas || []}
        onCreateArea={handleCreateArea}
        activeTasksCount={activeTasksCount}
      />
      
      <main className="flex-1 md:ml-64 relative h-full overflow-hidden flex flex-col transition-all">
        {renderPage()}
      </main>

      {/* Modals */}
      <TaskModal 
        task={editingTask}
        isOpen={!!editingTask}
        onClose={() => setEditingTask(null)}
        onSave={handleSaveTask}
        areas={appData?.areas || []}
      />
      <NoteModal 
        note={editingNote}
        isOpen={!!editingNote}
        onClose={() => setEditingNote(null)}
        onSave={handleSaveNote}
        areas={appData?.areas || []}
      />
      
      {/* Mobile nav toggle placeholder */}
      <button className="md:hidden fixed bottom-6 right-6 bg-primary text-white w-14 h-14 rounded-full shadow-lg flex items-center justify-center z-50">
         <span className="material-icons">menu</span>
      </button>
    </div>
  );
};
export default App;